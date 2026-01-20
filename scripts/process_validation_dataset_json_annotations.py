# %%
import os
from pathlib import Path
import json
import multiprocessing

from dotenv import load_dotenv
import pandas as pd
import pandas_gbq as pgb
from lingua import LanguageDetectorBuilder
import numpy as np
from joblib import Parallel, delayed

from qurating.constants import VALIDATION_DATA_CONFIG_DIR, VALIDATION_DATA_DATASETS_DIR
from qurating.validation.utils import process_resource
from qurating.validation.queries import QUERIES

load_dotenv(override=True)

HERE = Path(__file__).resolve().parent

DL = True
NJOBS = 300

ED_LEVEL_NUMERICAL = {
    "Preschool": 0,
    "Lower primary": 2,
    "Upper primary": 5,
    "Lower secondary": 8,
    "Upper secondary": 11,
    "Tertiary": 14,
}

BAD_RESOURCE_URLS = [
    "https://fabcontentcurationextsa.blob.core.windows.net/geeky-content/bloomlibrary-org/f3mfFT2C4A/pdf-document/e0b6b4e0b794e0b6bbe0b78fe0b6ab20e0b6b8e0b78fe0b6bbe0b78ae0b69ce0b6ba.pdf"
]

detector = LanguageDetectorBuilder.from_all_languages().build()

# %%
"""
Load metadata from GBQ with json annotations.
"""

query = QUERIES["json_annotated_gcs"]

df = pgb.read_gbq(query, project_id="fab-playground")
df = df.dropna(subset="gcs_url")
bad_docs = df["resource_url"].isin(BAD_RESOURCE_URLS).to_numpy()
df = df.loc[~bad_docs]

# %%
"""
Standardize the values for education level and material type using
education_level_mapping.xlsx and material_type_mapping.xlsx from
validation data config directory.

After this, keep only the rows where the language is English and
the standarized education level is not null.
"""

level_map_file = VALIDATION_DATA_CONFIG_DIR / "education_level_mapping.xlsx"
mat_type_map_file = VALIDATION_DATA_CONFIG_DIR / "material_type_mapping.xlsx"


level_map = pd.read_excel(level_map_file)
mat_type_map = pd.read_excel(mat_type_map_file)

level_map_dict = level_map.set_index("education_level")[
    "education_level_normalized"
].to_dict()
level_map_dict_isced = level_map.set_index("education_level")["isced level"].to_dict()
mat_type_map_dict = mat_type_map.set_index("material_type")[
    "material_type_normalized"
].to_dict()


def normalize_strings(level, mapping, split=False):
    if not isinstance(level, str):
        return level
    if split:
        out_level = []
        for level_ in level.split(","):
            level_ = level_.strip()
            out_level.append(float(mapping.get(level_, np.nan)))
        return out_level
    else:
        if level in mapping:
            return mapping[level]
        # else:
        #     print(level)


df["education_level_normalized"] = df["education_level"].apply(
    normalize_strings, args=(level_map_dict, False)
)
df["education_level_isced"] = df["education_level"].apply(
    normalize_strings, args=(level_map_dict_isced, False)
)
df["material_type_normalized"] = df["material_type"].apply(
    normalize_strings, args=(mat_type_map_dict, False)
)

usedf = df.dropna(subset=["education_level_normalized"])

print(len(usedf))

usedf.iloc[np.random.permutation(len(usedf))][
    [
        "education_level",
        "education_level_normalized",
        "education_level_isced",
        "material_type",
        "material_type_normalized",
    ]
].head(25)

# %%
"""
Print out the raw meta data for the first instance of each unique data source.
This is used to help produce bottum_up_meta_fields.yaml but serves no other
purpose.
"""

for key, val in (
    df.groupby("data_source")
    .agg("first")["raw"]
    .apply(lambda x: json.loads(x))
    .to_dict()
    .items()
):
    print(f"\n\n{'*'*50}\n{key}\n\n{json.dumps(val, indent=2)}\n{'#'*50}\n")

# %%
"""

"""
ncpus = multiprocessing.cpu_count()
print(ncpus)

BACKEND = "loky"

azure_storage_key = os.getenv("AZURE_STORAGE_KEY")

# drop any rows with missing values
subset = ["resource_url"]
usedf = usedf.dropna(subset=subset)

urls = usedf["resource_url"]
print(f"Found {len(urls)} unique URLs to download")

# Azure client kwargs
client_kwargs = {
    "account_url": "https://fabcontentcurationextsa.blob.core.windows.net",
    "credential": azure_storage_key,
}

rng = np.random.default_rng()
samp_i = rng.permutation(len(usedf))[:500]
samp_urls = urls.iloc[samp_i]

p = Parallel(
    n_jobs=NJOBS if BACKEND != "loky" else max(ncpus - 2, 1),
    backend=BACKEND,
    verbose=60,
)

res = p(
    delayed(process_resource)(
        url,
        scheme="az",
        dl=DL,
        verbose=0,
        client_kwargs=client_kwargs,
        extract_images=False,
    )
    for url in urls
)

# %%
pages_text, pages_images, sz = list(zip(*res))

full_text = ["\n".join(pt) for pt in pages_text]

usedf["full_text"] = full_text
usedf["nchars"] = usedf["full_text"].str.len()

usedf_hastext = usedf.loc[usedf["nchars"] >= 400].copy()

full_text = usedf_hastext["full_text"].to_list()

print(len(full_text))

# %%
# res = []
# with ThreadPoolExecutor(max_workers=NJOBS) as executor:
#     futures = []
#     for i, url in enumerate(tqdm(urls)):
#         future = executor.submit(
#             process_resource,
#             url,
#             scheme="az",
#             dl=DL,
#             verbose=0,
#             client_kwargs=client_kwargs,
#             i=i,
#         )
#         # res.append(future.result())
#         futures.append(future)
#     for future in tqdm(as_completed(futures), total=len(futures)):
#         res.append(future.result())

# %%
# md_text = pymupdf4llm.to_markdown(
#     local_path, write_images=True, image_path=Path("temp_images")
# )
# Path("output.md").write_bytes(md_text.encode())

# %%
all_result = detector.compute_language_confidence_values_in_parallel(full_text)

result_lang = [result[0].language.name for result in all_result]

usedf_hastext["lingua_language"] = result_lang
# for result, text in zip(all_result, full_text):
#     show_result = [r for r in result if r.value > 1e-3]
#     mid = len(text) // 2
#     print(f"\n{'*'*50}\n{show_result}:\n{'*'*50}\n\n" f"'{text[mid-100:mid+99]}'")

# %%
json_english = (usedf_hastext["language"].str.upper() == "ENGLISH").to_numpy()
json_not_english = (
    (usedf_hastext["language"].str.upper() != "ENGLISH")
    & ~(usedf_hastext["language"].isna())
).to_numpy()

print((usedf_hastext.loc[json_english, "lingua_language"] == "ENGLISH").mean())
print((usedf_hastext.loc[json_not_english, "lingua_language"] == "ENGLISH").mean())

print(
    usedf_hastext.loc[json_english]
    .groupby("lingua_language")
    .agg({"lingua_language": len})
)
print(
    usedf_hastext.loc[json_not_english]
    .groupby("lingua_language")
    .agg({"lingua_language": len})
)

# %%
lingua_english = usedf_hastext["lingua_language"] == "ENGLISH"

print(usedf_hastext.loc[lingua_english].groupby("language").agg({"language": len}))
# %%
usedf_hastext.loc[lingua_english].query("language == 'Tok Pisin'")[
    "full_text"
].to_list()


# %%
usedf_lingua_english = usedf_hastext.loc[lingua_english].reset_index()
usedf_lingua_english.to_parquet(
    VALIDATION_DATA_DATASETS_DIR / "bottom_up_sample_english.parquet"
)
