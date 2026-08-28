import logging
import re
from typing import List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ParsedTable():
    """
    Class for representing a table.
    Provides methods to access and format table data.
    """
    data: List[List[str]]
    num_header_rows: int = 0
    num_index_cols: int = 0
    infer_headers: bool = False
    max_header_rows: int = 5
    max_cols: int = 10

    def __post_init__(self):
        """Process table data after initialization."""
        self.num_rows = 0
        self.num_cols = 0
        self.headers = []
        self.indices = []

        if self.data:
            self.data = self._clean_cell_data(self.data)
            self.num_rows = len(self.data)
            self.num_cols = len(self.data[0])

            if (self.num_header_rows > 0):
                self.headers = self.data[:self.num_header_rows]
                self.headers = [self._fill_gaps(row) for row in self.headers]
            else:
                self.infer_headers = True

            if self.infer_headers:
                num_header_rows, headers = self._extract_headers(self.data)
                self.headers = headers
                self.num_header_rows = num_header_rows

            # default is that the first column is the index
            self.indices = [[row[0] for row in self.data]]
            self.num_index_cols = len(self.indices)
            self.header_names = self._format_multi_row_headers()

    def _extract_headers(self, raw_data: List[List[str]]):
        # Heaurstic for the top-k row count as headers
        # 1. iterate over each row from the top
        # 2. if the entire row is empty, skip it
        # 3. the row has 'gaps', then duplicate the value of the previous non-empty cell to fill the gap
        # 4. each time combine all the rows from the top and stop when there are no duplicates (except empty cells from the left)
        logger.debug("Inferring headers and indices for the table")
        multi_headers = []  # list of rows
        for i, row in enumerate(raw_data[:min(self.max_header_rows, len(raw_data))]):
            filled_row = self._fill_gaps(row)
            multi_headers.append(filled_row)
            logger.debug(f"Added header at row {i}: {row}")
            # if the whole row is empty, continue
            if all(not cell.strip() for cell in filled_row):
                continue
            # if the row has no gaps from the right, then we have found all the headers
            if filled_row == row:
                logger.debug(f"Done inferring headers at row {i} due to no gaps from the right")
                break
            # if there are no consecutive duplicates from the right, then we have found all the headers
            if self._no_consecutive_dup_cols(multi_headers):
                logger.debug(f"Done inferring headers at row {i} due to no gaps from the right")
                break

        num_header_rows = i + 1
        return num_header_rows, multi_headers
    
    def is_table_valid(self):
        """Check if the table is valid"""
        if self.num_header_rows >= self.max_header_rows:
            logger.warning(f"[Invalid Table] Table has too many headers (no less than {self.max_header_rows} rows).")
            return False 
           
        # check if the last header are all empty strings
        if all(not header.strip() for header in self.headers[-1]):
            logger.warning(f"[Invalid Table] Table has too many empty headers.")
            return False
        
        if self.num_cols > self.max_cols:
            logger.warning(f"[Invalid Table] Table has too many columns (more than {self.max_cols}).")
            return False
            
        if self.num_rows <= self.num_header_rows:
            logger.warning(f"[Invalid Table] Table has too few rows (no less than {self.num_header_rows} rows).")
            return False
        
        if (self.num_rows <= 1) or (self.num_cols <= 1):
            logger.warning(f"[Invalid Table] Table has too few rows or columns (no less than 1 row or column).")
            return False
            
        return True

    def get_standard_description(self):
        """Get the standard description of the table"""
        if len(self.headers) == 0: # no headers found
            return "This is an empty table."
        else:
            # use the last row of the multi-header as the col names
            # use the last column of the multi-index as the rows names
            col_names_str = ", ".join(self._get_last_header_row())
            row_names_str = ", ".join(self._get_last_index_col())
            return f"This table here describes what each of the {row_names_str} are for {col_names_str}, respectively."

    def describe_headers(self):
        """Describe the headers in a human readable format"""
        if len(self.headers) == 0:
            return "This is from a table with no headers."
        else:
            # if any of the headers are empty, then we should not include them in the description
            non_empty_headers = [header for header in self.header_names if header.strip()]
            header_string = ", ".join(non_empty_headers)
            return f"This is from a table with headers: {header_string}."

    def describe_table_row_by_row(self):
        """Describe the table row by row"""
        descriptions = ""
        if len(self.headers) == 0:
            descriptions += "This table has no headers."
        else:
            descriptions += self.describe_headers() + "\n"
            row_descriptions = []
            for i in range(self.num_rows):
                if i < self.num_header_rows:
                    continue
                else:
                    row_index = i - self.num_header_rows
                    row_data_as_string = self._format_row_data_as_string(self.data[i])
                    row_data_as_string = row_data_as_string.replace("\n", " ")
                    row_data_as_string = f"Row: {row_index + 1}, {row_data_as_string}"
                    row_descriptions.append(row_data_as_string)
            if len(row_descriptions) > 0:
                descriptions += "\n".join(row_descriptions)
        return descriptions

    def _format_row_data_as_string(self, row_data: List[str]) -> str:
        """Format the row data as a string with the header names"""
        # The format should be like: {header_name1}: {row_data1}; ..., {header_name2}: {row_data2}.
        pairs = zip(self.header_names, row_data)
        # if the the header name is empty, then just print the row data alone without the column name or :
        formatted_pairs = [f"{header}: {cell}" if header.strip() else f"{cell}" for header, cell in pairs]
        formatted_row_string = ", ".join(formatted_pairs) + "."
        return formatted_row_string

    def _get_last_header_row(self):
        """Get the last row of the multi-header and skip the index columns"""
        return self.headers[-1][self.num_index_cols:]

    def _get_last_index_col(self):
        """Get the last column of the multi-index and skip the multi-header"""
        return self.indices[-1][self.num_header_rows:]


    def _no_consecutive_dup_cols(self, potential_headers: List[List[str]]) -> bool:
        # return True if there are no consecutive duplicate columns from the right
        # Do not count any empty columns from the left
        # Transpose to get columns instead of rows
        columns = list(zip(*potential_headers))
        start_comparison = False
        for col_index in range(len(columns) -1):
            # if the current column is all empty, skip it
            if all(not cell.strip() for cell in columns[col_index]) and (not start_comparison):
                continue
            else:
                start_comparison = True
            if start_comparison and columns[col_index] == columns[col_index + 1]:
                return False
        return True


    def _format_multi_row_headers(self) -> List[str]:
        """Format the potential headers into a list of headers"""
        header_separator = ", "
        multi_header_joiner = " "
        transposed_headers = list(zip(*self.headers))
        # merge the multiple headers
        formatted_columns = []
        for multi_header in transposed_headers:
            formatted_column = multi_header_joiner.join([f"{header}" for header in multi_header])
            # remove the leading and trailing spaces and replace the header_separator with a space
            formatted_column = formatted_column.strip()
            formatted_column = formatted_column.replace(header_separator, multi_header_joiner)
            formatted_columns.append(formatted_column)
        return formatted_columns

    def _fill_gaps(self, row: List[str]) -> List[str]:
        """Fill in the gaps in the row from left to right"""
        filled_row = []
        found_non_empty_cell = False
        for cell in row:
            if cell.strip():
                found_non_empty_cell = True
            if found_non_empty_cell:
                if cell.strip():
                    filled_row.append(cell)
                else:
                    filled_row.append(filled_row[-1])
            else:
                filled_row.append(cell)
        return filled_row

    def _clean_cell_data(self, raw_data: List[List[str]]):
        """Replace None values with empty strings and ensure all cells are strings"""
        raw_row_cnt = len(raw_data)
        raw_col_cnt = max([len(row) for row in raw_data]) if raw_data else 0
        self.num_rows = raw_row_cnt
        self.num_cols = raw_col_cnt

        # First, ensure all rows have the same number of columns by padding with empty strings
        for i in range(raw_row_cnt):
            while len(raw_data[i]) < raw_col_cnt:
                raw_data[i].append("")

        for i in range(raw_row_cnt):
            for j in range(raw_col_cnt):
                if raw_data[i][j] is None:
                    raw_data[i][j] = ""
                else:
                    raw_data[i][j] = str(raw_data[i][j])
                # Remove leading and trailing whitespace
                raw_data[i][j] = raw_data[i][j].strip()
                # Replace consecutive whitespace with a single space
                raw_data[i][j] = re.sub(r'\s+', ' ', raw_data[i][j])
                # Replace newlines with a single space
                raw_data[i][j] = re.sub(r'\n', ' ', raw_data[i][j])
                # Replace dividers with a single space
                raw_data[i][j] = re.sub(r'\|', ' ', raw_data[i][j])
        return raw_data

    def write_as_markdown(self) -> str:
        """ Write the tables as a markdown table"""
        return self._format_table_data_as_markdown(self.data, num_header_rows=self.num_header_rows)

    def write_as_string(self) -> str:
        """ Write the tables as a tab-separated string"""
        return self._format_matrix_as_string(self.data)

    def write_transposed_as_markdown(self) -> str:
        """ Write the transposed table as a markdown table"""
        transposed_data = list(zip(*self.data))
        return self._format_table_data_as_markdown(transposed_data, num_header_rows=1)

    def write_transposed_as_string(self) -> str:
        """ Write the transposed table as a tab-separated string"""
        transposed_data = list(zip(*self.data))
        return self._format_matrix_as_string(transposed_data)

    def _format_matrix_as_string(self, matrix: List[List[str]]) -> str:
        result = ""
        for row in matrix:
            result += "\t".join([str(cell) for cell in row]) + "\n"
        return result

    def _format_table_data_as_markdown(self,
                                       matrix: List[List[str]],
                                       num_header_rows: int = 1) -> str:
        """Format the table data as a markdown table"""
        result = ""
        # Create header rows based on the number of header rows
        for i in range(self.num_rows):
            if num_header_rows == i:
                # Create separator row between the header and the data
                result += "|" + "|".join("---" for _ in matrix[0]) + "|\n"
            result += "| " + " | ".join(matrix[i]) + " |\n"
        return result

def get_parsed_table_from_md_format(markdown_string: str) -> ParsedTable:
    """Get a parsed table from a markdown format"""
    lines = [line.strip() for line in markdown_string.strip().split('\n')]
    lines = [line for line in lines if line]  # Remove empty lines

    if not lines or len(lines) < 3:
        return ParsedTable([])

    # Split each line into cells and clean them
    data = []
    for line in lines:
        if '|-' in line:  # Skip separator row
            continue
        # Remove leading/trailing |, split by |, and strip each cell
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        data.append(cells)

    # Count header rows by finding separator row
    num_header_rows = 0
    for i, line in enumerate(lines):
        if '|-' in line:
            num_header_rows = i
            break

    return ParsedTable(data, num_header_rows=num_header_rows, num_index_cols=1, infer_headers=False)
