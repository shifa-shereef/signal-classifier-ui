import * as XLSX from "xlsx";

export interface ParsedExcelData {
  rawData: number[][];
  columns: string[];
  signalData: number[][];
  signalColumns: string[];
}

// Expected column names (in order)
const EXPECTED_COLUMNS = [
  "Time - Current",
  "Current (kA) - Current",
  "Time - Voltage", 
  "Voltage (V) - Voltage",
  "Time - Resistance",
  "Resistance (Ohm) - Resistance",
  "Time - Force",
  "Force (kg) - Force",
];

// Signal column indices (the actual signal values, not time columns)
const SIGNAL_INDICES = [1, 3, 5, 7]; // Current, Voltage, Resistance, Force

const SIGNAL_NAMES = ["Current (kA)", "Voltage (V)", "Resistance (Ohm)", "Force (kg)"];

export async function parseExcelFile(file: File): Promise<ParsedExcelData> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const data = e.target?.result;
        const workbook = XLSX.read(data, { type: "binary" });
        
        // Get the first sheet
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        
        // Convert to JSON array
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { 
          header: 1,
          defval: 0 
        }) as (string | number)[][];

        if (jsonData.length < 2) {
          throw new Error("Excel file appears to be empty or has no data rows");
        }

        // Extract header row
        const headerRow = jsonData[0] as string[];
        
        // Validate we have 8 columns
        if (headerRow.length < 8) {
          throw new Error(`Expected 8 columns, found ${headerRow.length}. Please check the Excel format.`);
        }

        // Extract data rows (skip header)
        const dataRows = jsonData.slice(1).map(row => {
          return row.slice(0, 8).map(cell => {
            const num = typeof cell === 'number' ? cell : parseFloat(String(cell));
            return isNaN(num) ? 0 : num;
          });
        }).filter(row => row.some(val => val !== 0)); // Filter out completely empty rows

        if (dataRows.length === 0) {
          throw new Error("No valid data rows found in the Excel file");
        }

        // Extract only the 4 signal columns (ignoring time columns)
        const signalData = dataRows.map(row => 
          SIGNAL_INDICES.map(idx => row[idx] || 0)
        );

        resolve({
          rawData: dataRows,
          columns: headerRow.slice(0, 8),
          signalData,
          signalColumns: SIGNAL_NAMES,
        });
      } catch (error) {
        reject(error);
      }
    };

    reader.onerror = () => {
      reject(new Error("Failed to read the file"));
    };

    reader.readAsBinaryString(file);
  });
}

export function validateDataShape(signalData: number[][]): { valid: boolean; message: string } {
  const rows = signalData.length;
  const cols = signalData[0]?.length || 0;

  if (cols !== 4) {
    return { 
      valid: false, 
      message: `Expected 4 signal columns, found ${cols}` 
    };
  }

  if (rows < 10) {
    return { 
      valid: false, 
      message: `Not enough data rows. Found ${rows}, need at least 10` 
    };
  }

  return { 
    valid: true, 
    message: `Data shape: ${rows} time steps × ${cols} features` 
  };
}
