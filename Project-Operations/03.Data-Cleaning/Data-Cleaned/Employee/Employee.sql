-- Step 1: Remove duplicates if any
-- Remove exact duplicate rows
DELETE FROM Employee
WHERE EmployeeID IN (
    SELECT EmployeeID
    FROM (
        SELECT EmployeeID, ROW_NUMBER() OVER(PARTITION BY EmployeeID ORDER BY EmployeeID) AS RowNum
        FROM Employee
    ) AS TempTable
    WHERE RowNum > 1
);

-- Step 2: Check for null or invalid values in critical columns
-- Replace nulls in numeric columns with 0 or appropriate default values
UPDATE Employee
SET Salary = COALESCE(Salary, 0),
    [DistanceFromHome_(KM)] = COALESCE([DistanceFromHome_(KM)], 0),
    Age = COALESCE(Age, 0)
WHERE Salary IS NULL OR [DistanceFromHome_(KM)] IS NULL OR Age IS NULL;

-- Replace nulls in string columns with 'Unknown'
UPDATE Employee
SET Gender = COALESCE(Gender, 'Unknown'),
    Department = COALESCE(Department, 'Unknown'),
    JobRole = COALESCE(JobRole, 'Unknown')
WHERE Gender IS NULL OR Department IS NULL OR JobRole IS NULL;

-- Step 3: Ensure proper data types
-- Convert HireDate to proper date format
ALTER TABLE Employee
ALTER COLUMN HireDate DATE;

-- Step 4: Rename columns for better clarity (if needed)
-- Example: Rename 'DistanceFromHome (KM)' to 'DistanceFromHome'
ALTER TABLE Employee
RENAME COLUMN [DistanceFromHome_(KM)] TO DistanceFromHome;

-- Step 5: Remove unnecessary columns
-- If columns like 'Ethnicity' or 'State' are not needed, drop them
ALTER TABLE Employee