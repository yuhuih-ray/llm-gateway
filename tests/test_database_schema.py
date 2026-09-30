from llm_gateway.models import UsageLog


def test_usage_logs_tenant_foreign_key_without_extra_indexes():
    table = UsageLog.__table__
    foreign_keys = list(table.c.tenant_id.foreign_keys)

    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "tenants.id"
    assert foreign_keys[0].ondelete == "RESTRICT"
    assert foreign_keys[0].constraint.name == "fk_usage_logs_tenant_id_tenants"
    assert not table.c.tenant_id.nullable
    assert not table.indexes
