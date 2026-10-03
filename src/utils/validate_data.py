
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToBeBetween,
    ExpectColumnPairValuesAToBeGreaterThanB,
)
from typing import Tuple, List
import pandas as pd




def validate_telco_data(df) -> Tuple[bool, List[str]]:

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    print("🔍 Starting data validation with Great Expectations...")

    # === SET UP CONTEXT, DATA SOURCE, ASSET, AND BATCH (replaces PandasDataset) ===
    context = gx.get_context()

    data_source = context.data_sources.add_pandas(name="telco_datasource")
    data_asset = data_source.add_dataframe_asset(name="telco_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("telco_batch_def")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # === BUILD THE EXPECTATION SUITE ===
    suite = gx.ExpectationSuite(name="telco_validation_suite")
    suite = context.suites.add(suite)

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")

    # Customer identifier must exist (required for business operations)
    suite.add_expectation(ExpectColumnToExist(column="customerID"))
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="customerID"))

    # Core demographic features
    suite.add_expectation(ExpectColumnToExist(column="gender"))
    suite.add_expectation(ExpectColumnToExist(column="Partner"))
    suite.add_expectation(ExpectColumnToExist(column="Dependents"))

    # Service features (critical for churn analysis)
    suite.add_expectation(ExpectColumnToExist(column="PhoneService"))
    suite.add_expectation(ExpectColumnToExist(column="InternetService"))
    suite.add_expectation(ExpectColumnToExist(column="Contract"))

    # Financial features (key churn predictors)
    suite.add_expectation(ExpectColumnToExist(column="tenure"))
    suite.add_expectation(ExpectColumnToExist(column="MonthlyCharges"))
    suite.add_expectation(ExpectColumnToExist(column="TotalCharges"))

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    # Gender must be one of expected values (data integrity)
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]))

    # Yes/No fields must have valid values
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"]))
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"]))
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"]))

    # Contract types must be valid (business constraint)
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(
            column="Contract",
            value_set=["Month-to-month", "One year", "Two year"],
        )
    )

    # Internet service types (business constraint)
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(
            column="InternetService",
            value_set=["DSL", "Fiber optic", "No"],
        )
    )

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    # Tenure must be non-negative (business logic - can't have negative tenure)
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="tenure", min_value=0))

    # Monthly charges must be positive (business logic - no free service)
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0))

    # Total charges should be non-negative (business logic)
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0))

    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")

    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120))

    # Monthly charges should be within reasonable business range
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200))

    # No missing values in critical numeric features
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="tenure"))
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))

    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    suite.add_expectation(
        ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,  # Allow 5% exceptions for edge cases
        )
    )

    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = batch.validate(suite)

    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    columns = {}

    for r in results.results:
        if not r.success:
            expectation_type = r.expectation_config.type
            failed_expectations.append(expectation_type)
            kwargs = r.expectation_config.kwargs
            columns[expectation_type] = kwargs.get('column')

            
    # Print validation summary
    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    print("failed columns : ", columns)
    return results.success, failed_expectations


# path_ = r"C:\Users\bodap\OneDrive\Desktop\projects\Telco-Customer-Churn-ML\data\raw\Telco-Customer-Churn.csv"

# import pandas as pd
# df = pd.read_csv(path_)
# validate_telco_data(df)






