import pandas as pd


class PassDetector:
    """
    Assigns sequential pass numbers to GaitRite data rows based on
    the 'FootFall Object #' column.

    A pass represents a complete foot traversal, typically from
    foot contact 1 → 2 → 1 again.

    Example:
        FootFall Object #: [1, 2, 3, 1, 2, 3]
        Computed Pass:   [1, 1, 1, 2, 2, 2]
    """

    def __init__(self, footfall_col: str = "FootFall Object #"):
        """
        Parameters:
        - footfall_col: name of the column in the DataFrame
                        representing foot contact events
        """
        self.footfall_col = footfall_col

    def assign_pass_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'Computed Pass' column to the DataFrame, numbering
        each pass sequentially.

        Logic:
        1. Iterate through each value in the footfall column
        2. If the value is NaN → mark pass as None
        3. If the footfall number resets to 1 → increment pass counter
        4. Keep track of previous footfall for comparison

        Parameters:
        - df: pd.DataFrame containing the footfall column

        Returns:
        - pd.DataFrame with new column 'Computed Pass'
        """
        df = df.copy()
        pass_numbers = []
        current_pass = 0
        prev_footfall = None

        for val in df[self.footfall_col]:
            # Skip NaN values
            if pd.isna(val):
                pass_numbers.append(None)
                continue

            val = int(val)

            # New pass starts when footfall resets to 1 (and prev_footfall exists)
            if val == 1 and prev_footfall is not None:
                current_pass += 1

            # First valid footfall sets current_pass to 1
            if prev_footfall is None:
                current_pass = 1

            pass_numbers.append(current_pass)
            prev_footfall = val

        df["Computed Pass"] = pass_numbers
        return df
