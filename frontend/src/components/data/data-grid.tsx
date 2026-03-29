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
  emptyMessage = "No rows available.",
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
          "surface rounded-[1.6rem] px-5 py-8 text-sm text-[var(--color-text-muted)]",
          className,
        )}
      >
        {emptyMessage}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "surface overflow-hidden rounded-[1.6rem] border border-white/8 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]",
        className,
      )}
    >
      <div
        className="overflow-auto"
        style={maxHeight ? { maxHeight } : undefined}
      >
        <table className="min-w-full border-collapse">
          <thead className="sticky top-0 z-10 bg-[linear-gradient(180deg,rgba(18,22,28,0.98),rgba(16,20,26,0.92))] backdrop-blur-xl">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className={cn(
                      "mono border-b border-white/8 text-left text-[11px] font-medium uppercase tracking-[0.24em] text-[var(--color-text-muted)]",
                      density === "compact" ? "px-4 py-3" : "px-4 py-3.5",
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
                className="group border-b border-white/6 transition odd:bg-white/[0.012] hover:bg-white/[0.03] last:border-b-0"
              >
                {row.getVisibleCells().map((cell) => (
                  <td
                    key={cell.id}
                    className={cn(
                      "align-top text-sm text-[var(--color-text-soft)] transition group-hover:text-[var(--color-text)]",
                      density === "compact" ? "px-4 py-3" : "px-4 py-3.5",
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
