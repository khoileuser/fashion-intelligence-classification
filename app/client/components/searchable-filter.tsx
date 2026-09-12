"use client"

import { useId, useRef, useState } from "react"
import Image from "next/image"
import { Check, ChevronDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from "@/components/ui/popover"
import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
} from "@/components/ui/command"
import { cn } from "@/lib/utils"

export type SearchableFilterProps = {
    label: string
    searchPlaceholder?: string
    value: string
    onValueChange: (value: string) => void
    options: { value: string; label: string; icon?: string }[]
    className?: string
}

export function SearchableFilter({
    label,
    searchPlaceholder = `Search ${label.toLowerCase()}...`,
    value,
    onValueChange,
    options,
    className,
}: SearchableFilterProps) {
    const [open, setOpen] = useState(false)
    const input = useRef<HTMLInputElement>(null)
    const listId = useId()
    const selectedOption = options.find((option) => option.value === value)
    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger
                render={<Button variant="outline" />}
                role="combobox"
                aria-label={label}
                aria-expanded={open}
                aria-haspopup="listbox"
                aria-controls={open ? listId : undefined}
                className={cn(
                    "w-full min-w-0 justify-between font-normal sm:w-64",
                    className,
                )}
            >
                <span className="flex min-w-0 items-center gap-2">
                    {selectedOption?.icon && (
                        <Image
                            src={selectedOption.icon}
                            alt=""
                            width={20}
                            height={20}
                            className="size-5 shrink-0 object-contain"
                        />
                    )}
                    <span className="truncate">
                        {selectedOption?.label ?? label}
                    </span>
                </span>
                <ChevronDown className="size-4 shrink-0 opacity-50" />
            </PopoverTrigger>
            <PopoverContent
                align="start"
                className="w-(--anchor-width) p-0"
                initialFocus={input}
            >
                <Command>
                    <CommandInput
                        ref={input}
                        placeholder={searchPlaceholder}
                        aria-label={searchPlaceholder}
                    />
                    <CommandList id={listId}>
                        <CommandEmpty>No matching options.</CommandEmpty>
                        <CommandGroup>
                            {options.map((option) => (
                                <CommandItem
                                    key={option.value}
                                    value={`option:${option.value}`}
                                    keywords={[option.label]}
                                    onSelect={() => {
                                        onValueChange(option.value)
                                        setOpen(false)
                                    }}
                                >
                                    <Check
                                        aria-hidden="true"
                                        className={
                                            value === option.value
                                                ? "opacity-100"
                                                : "opacity-0"
                                        }
                                    />
                                    {option.icon && (
                                        <Image
                                            src={option.icon}
                                            alt=""
                                            width={20}
                                            height={20}
                                            className="size-5 shrink-0 object-contain"
                                        />
                                    )}
                                    {option.label}
                                </CommandItem>
                            ))}
                        </CommandGroup>
                    </CommandList>
                </Command>
            </PopoverContent>
        </Popover>
    )
}
