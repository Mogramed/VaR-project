declare module "@tanstack/react-table" {
  import type { ReactNode } from "react";

  export type HeaderContext<TData extends object> = {
    column: { columnDef: ColumnDef<TData> };
  };

  export type CellContext<TData extends object> = {
    row: { original: TData };
    column: { columnDef: ColumnDef<TData> };
  };

  export type ColumnDef<TData extends object> = {
    accessorKey?: keyof TData | string;
    id?: string;
    header?: ReactNode | ((context: HeaderContext<TData>) => ReactNode);
    cell?: ReactNode | ((context: CellContext<TData>) => ReactNode);
  };

  export type Header<TData extends object> = {
    id: string;
    isPlaceholder?: boolean;
    column: { columnDef: ColumnDef<TData> };
    getContext(): HeaderContext<TData>;
  };

  export type HeaderGroup<TData extends object> = {
    id: string;
    headers: Header<TData>[];
  };

  export type Cell<TData extends object> = {
    id: string;
    column: { columnDef: ColumnDef<TData> };
    getContext(): CellContext<TData>;
  };

  export type Row<TData extends object> = {
    id: string;
    original: TData;
    getVisibleCells(): Cell<TData>[];
  };

  export function flexRender<TProps>(
    renderer: ReactNode | ((props: TProps) => ReactNode),
    props: TProps,
  ): ReactNode;

  export function getCoreRowModel(): () => unknown;

  export function useReactTable<TData extends object>(options: {
    data: TData[];
    columns: ColumnDef<TData>[];
    getCoreRowModel: unknown;
  }): {
    getHeaderGroups(): HeaderGroup<TData>[];
    getRowModel(): { rows: Row<TData>[] };
  };
}

declare module "class-variance-authority" {
  type VariantValues = Record<string, Record<string, string>>;
  type VariantSelection<TVariants extends VariantValues> = {
    [K in keyof TVariants]?: keyof TVariants[K] | null | undefined;
  };

  export type VariantProps<T> =
    T extends (props?: infer P) => string ? NonNullable<P> : never;

  export function cva<TVariants extends VariantValues = Record<string, Record<string, string>>>(
    base: string,
    options?: {
      variants?: TVariants;
      defaultVariants?: {
        [K in keyof TVariants]?: keyof TVariants[K] | undefined;
      };
    },
  ): (props?: VariantSelection<TVariants>) => string;
}

declare module "react-markdown" {
  import type { ComponentType, ReactNode } from "react";

  export type MarkdownComponentProps = {
    children?: ReactNode;
  };

  export type Components = Record<string, ComponentType<MarkdownComponentProps>>;

  const Markdown: ComponentType<{
    children?: string;
    remarkPlugins?: unknown[];
    components?: Components;
  }>;

  export default Markdown;
}

declare module "remark-gfm" {
  const remarkGfm: unknown;
  export default remarkGfm;
}

declare module "tailwind-merge" {
  export function twMerge(
    ...classLists: Array<string | null | undefined | false>
  ): string;
}
