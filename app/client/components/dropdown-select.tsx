"use client"

import { Children, isValidElement, type ReactNode } from "react"
import { SearchableFilter } from "@/components/searchable-filter"

type OptionProps = { value: string; children: ReactNode }
export function DropdownOption({ children }: OptionProps) {
    return <>{children}</>
}

function optionText(node: ReactNode): string {
    return Children.toArray(node)
        .map((child) =>
            isValidElement<{ children?: ReactNode }>(child)
                ? optionText(child.props.children)
                : String(child),
        )
        .join("")
}

/** Compatibility layout for existing filter and attribute option declarations. */
export function DropdownSelect({
    value,
    onValueChange,
    children,
    className,
    "aria-label": label,
}: {
    value: string
    onValueChange: (value: string) => void
    children: ReactNode
    className?: string
    "aria-label": string
}) {
    const options = Children.toArray(children).flatMap((child) =>
        isValidElement<OptionProps>(child)
            ? [
                  {
                      value: child.props.value,
                      label: optionText(child.props.children),
                  },
              ]
            : [],
    )
    return (
        <SearchableFilter
            label={label}
            value={value}
            onValueChange={onValueChange}
            options={options}
            className={className}
        />
    )
}
