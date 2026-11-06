"""SFT data: source JSON -> {question, answer} -> token bins.

Special tokens in the packed sequence are

    <BOS> <USR> {question} <ASSISTANT> {answer} <END>

Their ids are passed to `tokenize` (the strings are not in the pretrain
tokenizer yet).

    # fido-ai UA-SQuAD -> text pairs
    python prepare_sft.py fido \\
      --train ~/data/fido-ai/train.json \\
      --val ~/data/fido-ai/val.json \\
      --out-dir $LLMUA_ROOT/sft_data/text --dry-run

    # fido-ua-news CSV -> article category pairs
    python prepare_sft.py news \\
      --train ~/data/fido-ua-news/train.csv \\
      --test ~/data/fido-ua-news/test.csv \\
      --out-dir $LLMUA_ROOT/sft_data/text --dry-run

    # UA-GEC fluency source/target sentences
    python prepare_sft.py gec \\
      --root ~/data/ua-gec/data/gec-fluency \\
      --out-dir $LLMUA_ROOT/sft_data/text --dry-run

    # Aya parquet, language_code=ukr only
    python prepare_sft.py aya \\
      --dir ~/data/CohereLabs/aya_dataset/data \\
      --out-dir $LLMUA_ROOT/sft_data/text --dry-run

    # tokenize: --val-nums  -p path,val_n[,train_n]  after shuffle+length filter
    python prepare_sft.py tokenize \\
      --val-nums --seed 0 \\
      -p $LLMUA_ROOT/sft_data/text/fidoai.json,100,500 \\
      -p $LLMUA_ROOT/sft_data/text/fidonews.json,200 \\
      -p $LLMUA_ROOT/sft_data/text/uagec.json,50,1000 \\
      -p $LLMUA_ROOT/sft_data/text/aya.json,20 \\
      --out $LLMUA_ROOT/sft_data/bins/32768 \\
      --bos-id 0 --usr-id 201 --assistant-id 202 --end-id 203 \\
      --context-width 2048 --dry-run
"""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        print("\ncommands: fido, news, gec, aya, tokenize"
              "  (try: python prepare_sft.py fido -h)")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "fido":
        from sft.from_fido import parse_args, run
    elif cmd == "news":
        from sft.from_news import parse_args, run
    elif cmd == "gec":
        from sft.from_gec import parse_args, run
    elif cmd == "aya":
        from sft.from_aya import parse_args, run
    elif cmd == "tokenize":
        from sft.tokenize import parse_args, run
    else:
        raise SystemExit(
            f"unknown command {cmd!r}; "
            "expected fido, news, gec, aya, or tokenize")
    run(parse_args(rest, prog=f"prepare_sft.py {cmd}"))


if __name__ == "__main__":
    main()
