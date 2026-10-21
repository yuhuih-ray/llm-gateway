import argparse
import asyncio
import getpass
import sys
from decimal import Decimal, InvalidOperation

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from llm_gateway.admin_auth import hash_password
from llm_gateway.auth import generate_key, hash_key, parse_key
from llm_gateway.auth_cache import Action, invalidate_cli_auth
from llm_gateway.db import get_engine, get_sessionmaker
from llm_gateway.models import ApiKey, Tenant, User


async def run(args: argparse.Namespace) -> None:  # Parsed CLI arguments.
    # Prompt and hash before opening a session; never put passwords in argv.
    password_hash = None
    if args.command == "create-user":
        password = getpass.getpass("Password: ")
        if not password:
            raise ValueError("Password must not be empty")
        password_hash = hash_password(password)
    invalidation: Action | None = None
    key_hashes: list[str] = []
    try:
        async with get_sessionmaker()() as session:
            async with session.begin():
                if args.command == "create-tenant":
                    session.add(Tenant(name=args.name))
                elif args.command in {
                    "set-budget",
                    "suspend-tenant",
                    "reactivate-tenant",
                }:
                    # Operator CLI only: tenant admins must not raise their own budgets.
                    tenant = await session.scalar(
                        select(Tenant).where(Tenant.name == args.tenant)
                    )
                    if tenant is None:
                        raise ValueError("Tenant not found")
                    if args.command == "set-budget":
                        tenant.monthly_budget_usd = parse_budget(args.usd)
                        invalidation = "clear"
                    else:
                        tenant.status = (
                            "suspended"
                            if args.command == "suspend-tenant"
                            else "active"
                        )
                        invalidation = (
                            "suspend" if args.command == "suspend-tenant" else "clear"
                        )
                    key_hashes = list(
                        await session.scalars(
                            select(ApiKey.key_hash).where(ApiKey.tenant_id == tenant.id)
                        )
                    )
                elif args.command == "create-key":
                    tenant = await session.scalar(
                        select(Tenant).where(Tenant.name == args.tenant)
                    )
                    if tenant is None:
                        raise ValueError("Tenant not found")
                    key = generate_key()
                    session.add(
                        ApiKey(
                            tenant_id=tenant.id,
                            name=args.name,
                            key_prefix=parse_key(key),
                            key_hash=hash_key(key),
                        )
                    )
                elif args.command == "create-user":
                    tenant = await session.scalar(
                        select(Tenant).where(Tenant.name == args.tenant)
                    )
                    if tenant is None:
                        raise ValueError("Tenant not found")
                    session.add(
                        User(
                            tenant_id=tenant.id,
                            email=args.email,
                            password_hash=password_hash,
                            role=args.role,
                        )
                    )
                else:
                    result = await session.execute(
                        update(ApiKey)
                        .where(ApiKey.key_prefix == args.prefix)
                        .values(revoked_at=func.now())
                        .returning(ApiKey.key_hash)
                    )
                    digest = result.scalar_one_or_none()
                    if digest is None:
                        raise ValueError("API key not found")
                    key_hashes = [digest]
                    invalidation = "revoke"
        # Invalidation is after commit AND session exit, never during DB work.
        if invalidation is not None:
            await invalidate_cli_auth(key_hashes, invalidation)
        if args.command == "create-key":
            print(
                "Save this API key now; it will not be shown again.",
                file=sys.stderr,
            )
            print(key)
        else:
            print(
                "Tenant created."
                if args.command == "create-tenant"
                else "User created."
                if args.command == "create-user"
                else "Budget updated."
                if args.command == "set-budget"
                else "Tenant suspended."
                if args.command == "suspend-tenant"
                else "Tenant reactivated."
                if args.command == "reactivate-tenant"
                else "API key revoked."
            )
    finally:
        await get_engine().dispose()


def parse_budget(value: str) -> Decimal | None:
    if value.lower() == "none":
        return None
    try:
        amount = Decimal(value)
        if (
            not amount.is_finite()
            or amount < 0
            or amount >= Decimal("10000000000")
            or amount != amount.quantize(Decimal("0.01"))
        ):
            raise ValueError(
                "Budget must be nonnegative USD with at most two decimal places"
            )
        return amount
    except InvalidOperation:
        raise ValueError("Invalid budget amount") from None


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage gateway tenants and API keys")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create-tenant").add_argument("name")
    budget = commands.add_parser("set-budget")
    budget.add_argument("--tenant", required=True)
    budget.add_argument("--usd", required=True)
    for command in ("suspend-tenant", "reactivate-tenant"):
        commands.add_parser(command).add_argument("--tenant", required=True)
    create = commands.add_parser("create-key")
    create.add_argument("--tenant", required=True)
    create.add_argument("--name", required=True)
    commands.add_parser("revoke-key").add_argument("prefix")
    user = commands.add_parser("create-user")
    user.add_argument("--tenant", required=True)
    user.add_argument("--email", required=True)
    user.add_argument("--role", required=True, choices=["admin", "member", "viewer"])
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    except IntegrityError:
        parser.exit(
            1,
            "Database constraint violation; tenant name, email, or key prefix may already exist.\n",
        )


if __name__ == "__main__":
    main()
