"use client";

import {
  flexRender,
  getCoreRowModel,
  useReactTable,
  type ColumnDef,
} from "@tanstack/react-table";
import { cn } from "@/lib/utils";

export function DataGrid<TData extends object>({
  data,
  columns,
  emptyMessage = "No data available.",
  maxHeight,
  density = "compact",
  className,
}: {
  data: TData[];
  columns: ColumnDef<TData>[];
  emptyMessage?: string;
  maxHeight?: string;
  density?: "compact" | "comfortable";
  className?: string;
}) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  if (data.length === 0) {
    return (
      <div
        className={cn(
          "rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-6 text-xs text-[var(--color-text-muted)]",
          className,
        )}
      >
        {emptyMessage}
      </div>
    );
  }

  const cellPad = density === "compact" ? "px-3 py-[7px]" : "px-3 py-2.5";

  return (
    <div
      className={cn(
        "overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)]",
        className,
      )}
    >
      <div
        className="overflow-auto"
        style={maxHeight ? { maxHeight } : undefined}
      >
        <table className="min-w-full border-collapse text-[12px]">
          <thead className="sticky top-0 z-10 bg-[var(--color-surface-strong)]">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className={cn(
                      "border-b border-[var(--color-border)] text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]",
                      cellPad,
                    )}
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className="border-b border-[var(--color-border)] transition-colors last:border-b-0 hover:bg-[var(--color-surface-hover)]"
              >
                {row.getVisibleCells().map((cell) => (
                  <td
                    key={cell.id}
                    className={cn(
                      "align-top text-[var(--color-text-soft)]",
                      cellPad,
                    )}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
