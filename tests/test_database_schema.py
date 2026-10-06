from llm_gateway.models import UsageLog


def test_usage_logs_tenant_foreign_key_and_report_index():
    table = UsageLog.__table__
    foreign_keys = list(table.c.tenant_id.foreign_keys)

    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "tenants.id"
    assert foreign_keys[0].ondelete == "RESTRICT"
    assert foreign_keys[0].constraint.name == "fk_usage_logs_tenant_id_tenants"
    assert not table.c.tenant_id.nullable
    assert len(table.indexes) == 1
    index = next(iter(table.indexes))
    assert index.name == "ix_usage_logs_tenant_id_created_at"
    assert list(index.columns.keys()) == ["tenant_id", "created_at"]
    assert not index.unique
    assert not index.dialect_options["postgresql"].get("include")
