def log_divider(wide=50, *, following_nl=False):
    if following_nl:
        print()
    print(
        "=" * wide,
    )


def log_header(header: str, *, wide=50):
    log_divider(wide=wide, following_nl=True)
    print("  " + header)
    log_divider(wide=wide, following_nl=False)
