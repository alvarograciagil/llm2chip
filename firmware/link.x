/* Program memory at address 0 and data memory on the internal bus. */
MEMORY
{
  PROGRAM (rx) : ORIGIN = 0x00000000, LENGTH = 256K
  DATA (rw) : ORIGIN = 0x10000000, LENGTH = 32K
}

ENTRY(_start)

SECTIONS
{
  .text : { KEEP(*(.text.entry)) *(.text .text.*) } > PROGRAM
  .rodata : { *(.rodata .rodata.*) *(.srodata .srodata.*) } > DATA
  .data : { *(.data .data.*) *(.sdata .sdata.*) } > DATA
  .bss (NOLOAD) : { __bss_start = .; *(.bss .bss.*) *(.sbss .sbss.*) . = ALIGN(4); __bss_end = .; } > DATA
  __stack_top = ORIGIN(DATA) + LENGTH(DATA);
  /DISCARD/ : { *(.comment) *(.note .note.*) *(.eh_frame .eh_frame_hdr) *(.riscv.attributes) }
}
