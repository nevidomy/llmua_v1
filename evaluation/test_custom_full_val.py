"""Custom full-val key helpers.

Run: python -m evaluation.test_custom_full_val
"""
from model.packed_val import custom_loss_key, is_custom_loss_key


def check(cond, label):
    assert cond, label
    print(f"  ok: {label}")


def test_keys():
    print("=== custom loss keys ===")
    check(custom_loss_key("pretrain", "15k") == "full_val_loss_15k",
          "pretrain stream key")
    check(custom_loss_key("pretrain", "bruk", "packed")
          == "full_val_loss_packed_bruk", "pretrain packed key")
    check(custom_loss_key("pretrain", "bruk_uk4b_sentenses", "packed_uk4b")
          == "full_val_loss_packed_uk4b_bruk_uk4b_sentenses",
          "packed_uk4b key")
    check(custom_loss_key("sft", "ua") == "full_val_loss_sft_ua", "sft key")
    check(custom_loss_key("sft", "ua", "packed") == "full_val_loss_sft_packed_ua",
          "sft packed key")


def test_is_custom():
    print("=== is_custom_loss_key ===")
    for k in ("full_val_loss_15k", "full_val_loss_packed_x",
              "full_val_loss_packed_uk4b_x", "full_val_loss_sft_ua"):
        check(is_custom_loss_key(k), f"accepts {k}")
    for k in ("full_val", "val_nats", "final", "", None):
        check(not is_custom_loss_key(k), f"rejects {k!r}")


def main():
    test_keys()
    test_is_custom()
    print("CUSTOM FULL VAL PASS")


if __name__ == "__main__":
    main()
