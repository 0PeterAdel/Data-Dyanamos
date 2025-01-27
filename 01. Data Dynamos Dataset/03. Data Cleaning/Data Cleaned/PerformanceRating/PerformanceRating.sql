SELECT * FROM my_database.performancerating;
-- Step 1: Remove duplicate records based on PerformanceID
DELETE FROM PerformanceRating
WHERE PerformanceID IN (
    SELECT PerformanceID
    FROM (
        SELECT PerformanceID, ROW_NUMBER() OVER (PARTITION BY PerformanceID ORDER BY ReviewDate) AS row_num
        FROM PerformanceRating
    ) AS temp
    WHERE row_num > 1
);

-- Step 2: Ensure ReviewDate is in a valid date format
-- Assuming the format is MM/DD/YYYY, convert invalid dates to NULL
UPDATE PerformanceRating
SET ReviewDate = NULL
WHERE TRY_CAST(ReviewDate AS DATE) IS NULL;

-- Step 3: Replace negative or out-of-range values with NULL for numeric columns
UPDATE PerformanceRating
SET EnvironmentSatisfaction = NULL
WHERE EnvironmentSatisfaction NOT BETWEEN 1 AND 5;

UPDATE PerformanceRating
SET JobSatisfaction = NULL
WHERE JobSatisfaction NOT BETWEEN 1 AND 5;

UPDATE PerformanceRating
SET RelationshipSatisfaction = NULL
WHERE RelationshipSatisfaction NOT BETWEEN 1 AND 5;

UPDATE PerformanceRating
SET WorkLifeBalance = NULL
WHERE WorkLifeBalance NOT BETWEEN 1 AND 5;

-- Step 4: Check for mismatched training data
-- TrainingOpportunitiesTaken should not exceed TrainingOpportunitiesWithinYear
UPDATE PerformanceRating
SET TrainingOpportunitiesTaken = TrainingOpportunitiesWithinYear
WHERE TrainingOpportunitiesTaken > TrainingOpportunitiesWithinYear;

-- Step 5: Add a column for Data Quality Status (optional)
ALTER TABLE PerformanceRating ADD DataQualityStatus VARCHAR(50);

UPDATE PerformanceRating
SET DataQualityStatus = CASE
    WHEN ReviewDate IS NULL THEN 'Invalid Date'
    WHEN EnvironmentSatisfaction IS NULL OR JobSatisfaction IS NULL OR RelationshipSatisfaction IS NULL THEN 'Missing Satisfaction Data'
    WHEN TrainingOpportunitiesTaken > TrainingOpportunitiesWithinYear THEN 'Training Mismatch'
    ELSE 'Valid'
END;
