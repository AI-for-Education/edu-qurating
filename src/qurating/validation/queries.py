QUERIES = {

    #####################################################################
    # View basic query with azure links
    "main_view": "SELECT * FROM `fab-playground.content_curation.fct_meta_and_annotations`",

    ######################################################################
    # Basic query with azure links
    "human_annotated": """
    WITH meta_urls AS (
    SELECT meta._id AS id, meta.website, meta.title, meta.status, meta.updatedAt, meta.raw, urls.file_ext, urls.url AS url, urls.content_category AS category
    FROM `fab-playground.content_curation.raw_meta_data` meta
    LEFT JOIN `fab-playground.content_curation.raw_urls` urls
    ON meta._id = urls._id
    ),
    selected_meta AS (
    SELECT *
    FROM meta_urls
    WHERE STARTS_WITH(url, 'https://fabcontentcurationextsa.blob.core.windows.net') AND (file_ext = 'pdf' OR file_ext = 'docx')
    )

    SELECT selected_meta.*, anno.* EXCEPT(id, category, `FLN suitability`), anno.`FLN suitability` AS FLN_suitability,

    FROM selected_meta
    LEFT JOIN `fab-playground.content_curation.raw_annotated_data` anno
    ON selected_meta.id = anno.id
    """,

    #####################################################################
    ###### With gcs storage links
    "human_annotated_gcs": r"""
    WITH urls_with_gcs_link AS (
    -- Construct GCS URL from urls parquet data in BigQuery
    SELECT
        *,
        CASE
        WHEN local_path IS NOT NULL AND NOT STARTS_WITH(url, 'sftp') THEN
            CONCAT('gs://fab-playground-bucket/cc_content_processed/', local_path)

        WHEN url LIKE '%blob.core.windows.net/geeky-content/%' THEN
            CONCAT(
            'gs://fab-playground-bucket/cc_content_processed/resources/',
            REGEXP_EXTRACT(url, r'blob\.core\.windows\.net/geeky-content/(.+)')
            )

        WHEN url LIKE '%blob.core.windows.net/merlot-content/%' THEN
            CONCAT(
            'gs://fab-playground-bucket/cc_content_processed/resources/merlot/',
            REGEXP_EXTRACT(url, r'blob\.core\.windows\.net/merlot-content/(.+)')
            )
        ELSE NULL
        END AS gcs_url
    FROM `fab-playground.content_curation.raw_urls`
    WHERE file_ext = 'pdf' OR file_ext = 'docx' OR file_ext = 'txt'
    ),
    meta_urls AS (
    SELECT meta._id AS id, meta.website, meta.title, meta.status, meta.updatedAt, urls.file_ext, urls.gcs_url, meta.raw, CONCAT(urls.url, '?', 'sp=rl&st=2025-12-03T14:58:44Z&se=2025-12-31T23:13:44Z&spr=https&sv=2024-11-04&sr=c&sig=l9Io%2BhgxjNmt2woeMCN7%2BMNCdaTpaK73jQqw86tmKmw%3D') AS url, urls.content_category AS category
    FROM `fab-playground.content_curation.raw_meta_data` meta
    LEFT JOIN urls_with_gcs_link urls
    ON meta._id = urls._id
    ),
    selected_meta AS (
    SELECT *
    FROM meta_urls
    WHERE STARTS_WITH(url, 'https://fabcontentcurationextsa.blob.core.windows.net') AND (file_ext = 'pdf' OR file_ext = 'docx')
    )

    SELECT selected_meta.*, anno.* EXCEPT(id, category, `FLN suitability`), anno.`FLN suitability` AS FLN_suitability,

    FROM selected_meta
    LEFT JOIN `fab-playground.content_curation.raw_annotated_data` anno
    ON selected_meta.id = anno.id
    """,


    ######################################################################
    ##### With json annotations
    "json_annotated_gcs": r"""
    WITH urls_with_gcs_link AS (
    -- Construct GCS URL from urls parquet data in BigQuery
    SELECT
        *,
        CASE
        -- Case 1: Has local_path (extracted files) - use it directly
        WHEN local_path IS NOT NULL AND NOT STARTS_WITH(url, 'sftp') THEN
            CONCAT('gs://fab-playground-bucket/cc_content_processed/', local_path)

        -- Case 2: Azure geeky-content URLs - extract blob path after container
        WHEN url LIKE '%blob.core.windows.net/geeky-content/%' THEN
            CONCAT(
            'gs://fab-playground-bucket/cc_content_processed/resources/',
            REGEXP_EXTRACT(url, r'blob\.core\.windows\.net/geeky-content/(.+)')
            )

        -- Case 3: Azure merlot-content URLs - prefix with resources/merlot/
        WHEN url LIKE '%blob.core.windows.net/merlot-content/%' THEN
            CONCAT(
            'gs://fab-playground-bucket/cc_content_processed/resources/merlot/',
            REGEXP_EXTRACT(url, r'blob\.core\.windows\.net/merlot-content/(.+)')
            )
        -- Case 4: Non-Azure URLs (S3, etc.) - can't construct GCS path
        ELSE NULL
        END AS gcs_url
    FROM `fab-playground.content_curation.raw_urls`
    WHERE file_ext = 'pdf' OR file_ext = 'docx' OR file_ext = 'txt'
    ),
    meta_urls AS (
    SELECT meta.* EXCEPT (_id), meta._id AS id, urls.file_ext, urls.gcs_url, urls.url AS resource_url, urls.content_category AS category,
    FROM `fab-playground.content_curation.fct_meta_data_enriched` meta
    LEFT JOIN urls_with_gcs_link urls
    ON meta._id = urls._id
    ),
    selected_meta AS (
    SELECT *
    FROM meta_urls
    WHERE (file_ext = 'pdf' OR file_ext = 'docx' OR file_ext = 'txt')
    )

    SELECT selected_meta.*, anno.* EXCEPT(id, category, `FLN suitability`, grade, student_teacher, subject, country, language), anno.`FLN suitability` AS FLN_suitability, anno.grade AS annotated_grade, anno.student_teacher AS annotated_student_teacher,
    anno.subject AS annotated_subject, anno.country AS annotated_country, anno.language AS annotated_language

    FROM selected_meta
    LEFT JOIN `fab-playground.content_curation.raw_annotated_data` anno
    ON selected_meta.id = anno.id
    """,
}