import { useMemo } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";

interface DataTableProps {
  data: number[][];
  columns: string[];
  maxRows?: number;
}

export function DataTable({ data, columns, maxRows = 10 }: DataTableProps) {
  const displayData = useMemo(() => {
    return data.slice(0, maxRows);
  }, [data, maxRows]);

  const formatValue = (value: number) => {
    if (typeof value !== 'number' || isNaN(value)) return '—';
    return value.toFixed(4);
  };

  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border/50 flex items-center justify-between">
        <h3 className="text-sm font-medium text-foreground">Signal Data Preview</h3>
        <span className="text-xs text-muted-foreground font-mono">
          {data.length} rows × {columns.length} columns
        </span>
      </div>
      
      <ScrollArea className="h-[400px]">
        <Table>
          <TableHeader className="sticky top-0 bg-card z-10">
            <TableRow className="border-border/50 hover:bg-transparent">
              <TableHead className="text-muted-foreground font-mono text-xs w-12">#</TableHead>
              {columns.map((col, i) => (
                <TableHead 
                  key={i} 
                  className="text-muted-foreground font-mono text-xs whitespace-nowrap"
                >
                  {col}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {displayData.map((row, rowIndex) => (
              <TableRow 
                key={rowIndex} 
                className="border-border/30 hover:bg-secondary/30 transition-colors"
              >
                <TableCell className="font-mono text-xs text-muted-foreground">
                  {rowIndex + 1}
                </TableCell>
                {row.map((cell, cellIndex) => (
                  <TableCell 
                    key={cellIndex} 
                    className="font-mono text-sm text-foreground/90"
                  >
                    {formatValue(cell)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </ScrollArea>
      
      {data.length > maxRows && (
        <div className="px-4 py-2 border-t border-border/50 text-center">
          <span className="text-xs text-muted-foreground">
            Showing {maxRows} of {data.length} rows
          </span>
        </div>
      )}
    </div>
  );
}
