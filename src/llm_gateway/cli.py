import argparse
import asyncio
import sys

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from llm_gateway.auth import generate_key, hash_key, parse_key
from llm_gateway.db import get_engine, get_sessionmaker
from llm_gateway.models import ApiKey, Tenant


async def run(args: argparse.Namespace) -> None:  # Parsed CLI arguments.
    try:
        async with get_sessionmaker()() as session:
            async with session.begin():
                if args.command == "create-tenant":
                    session.add(Tenant(name=args.name))
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
                else:
                    result = await session.execute(
                        update(ApiKey)
                        .where(ApiKey.key_prefix == args.prefix)
                        .values(revoked_at=func.now())
                        .returning(ApiKey.id)
                    )
                    if result.scalar_one_or_none() is None:
                        raise ValueError("API key not found")
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
                    else "API key revoked."
                )
    finally:
        await get_engine().dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage gateway tenants and API keys")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create-tenant").add_argument("name")
    create = commands.add_parser("create-key")
    create.add_argument("--tenant", required=True)
    create.add_argument("--name", required=True)
    commands.add_parser("revoke-key").add_argument("prefix")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    except IntegrityError:
        parser.exit(
            1,
            "Database constraint violation; tenant name or key prefix may already exist.\n",
        )


if __name__ == "__main__":
    main()
