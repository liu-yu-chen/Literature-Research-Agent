import json
from collections import Counter
from pathlib import Path


FILE_PATH = "filtered.json"


def inspect_json(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"File not found: {file_path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print("Basic Information")
    print("=" * 60)

    print("JSON Type:", type(data).__name__)

    if isinstance(data, list):
        print("Number of Records:", len(data))

        if len(data) == 0:
            return

        sample = data[0]

    elif isinstance(data, dict):
        print("Number of Keys:", len(data))
        print("Top Keys:", list(data.keys())[:20])

        sample = data

    else:
        print("Unsupported JSON structure")
        return


    print("\n" + "=" * 60)
    print("Sample Record")
    print("=" * 60)

    print(json.dumps(
        sample,
        indent=2,
        ensure_ascii=False
    )[:3000])


    if not isinstance(data, list):
        return


    print("\n" + "=" * 60)
    print("Field Analysis")
    print("=" * 60)


    all_fields = Counter()

    for item in data:
        if isinstance(item, dict):
            all_fields.update(item.keys())


    print("\nField Frequency:")
    for field, count in all_fields.most_common():
        print(
            f"{field:<40} {count}/{len(data)} "
            f"({count/len(data)*100:.2f}%)"
        )


    print("\n" + "=" * 60)
    print("Missing Field Analysis")
    print("=" * 60)


    fields = list(all_fields.keys())

    for field in fields:
        missing = sum(
            1 for item in data
            if not isinstance(item, dict)
            or field not in item
            or item[field] is None
        )

        if missing:
            print(
                f"{field:<40} missing: {missing} "
                f"({missing/len(data)*100:.2f}%)"
            )


    print("\n" + "=" * 60)
    print("Value Type Analysis")
    print("=" * 60)


    for field in fields:

        types = Counter()

        for item in data:

            if isinstance(item, dict) and field in item:

                value = item[field]

                if value is None:
                    types["None"] += 1

                else:
                    types[type(value).__name__] += 1


        print(f"\n{field}")
        for t, c in types.items():
            print(
                f"  {t}: {c}"
            )


    print("\n" + "=" * 60)
    print("Nested Structure Examples")
    print("=" * 60)


    for field in fields:

        example = None

        for item in data:

            if isinstance(item, dict) and field in item:
                if isinstance(item[field], (list, dict)):
                    example = item[field]
                    break


        if example is not None:

            print("\nField:", field)

            print(
                json.dumps(
                    example,
                    indent=2,
                    ensure_ascii=False
                )[:1000]
            )


if __name__ == "__main__":
    inspect_json(FILE_PATH)