from src.common.schema import is_valid_record, validate_record

VALID = {
    "doc_type": "e_invoice",
    "seller_name": "統一超商",
    "date": "2024-05-12",
    "invoice_number": "AB12345678",
    "seller_tax_id": "22555003",
    "buyer_tax_id": None,
    "total_amount": 1234,
    "items": [{"name": "拿鐵咖啡", "amount": 60}],
}


def test_valid_record_passes():
    assert is_valid_record(VALID)


def test_nullable_fields():
    rec = {
        **VALID,
        "seller_name": None,
        "date": None,
        "invoice_number": None,
        "total_amount": None,
    }
    assert is_valid_record(rec)


def test_missing_key_fails():
    rec = {k: v for k, v in VALID.items() if k != "seller_tax_id"}
    errors = validate_record(rec)
    assert errors and "seller_tax_id" in errors[0]


def test_bad_invoice_number_pattern_fails():
    assert not is_valid_record({**VALID, "invoice_number": "A12345678"})


def test_extra_key_fails():
    assert not is_valid_record({**VALID, "hallucinated_field": 1})


def test_item_requires_name_and_amount():
    assert not is_valid_record({**VALID, "items": [{"name": "咖啡"}]})
