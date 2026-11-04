import json, sys
want = ["newspaper_0b1bb8d03b4287eb95f67b68c2cf9f92_1", "newspaper_ef5c785d9cc21ccf6b73d143f231b5c5_1",
        "newspaper_e3842a5681ef54c62a58804adec06c3f_1", "page-dc728bd5-50a9-41e8-9e5a-d4b01e6664a2",
        "page-b2cc5729-a6b2-49ba-af62-788bf069b617", "newspaper_01f020903d7bc4b3fc69f500f41eb77e_1",
        "scihub_s00381-021-05138-3.pdf_3", "newspaper_b615618f3cdb0f2c13ffb5865dea799c_1"]
items = {i["id"]: i for i in json.load(open(sys.argv[1]))["single_pages"]}
for w in want:
    it = items["omni-" + w]
    print(it["id"], it["image_path"])
