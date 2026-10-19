cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tsr_uart_inference_esp32s3)

target_sources(app PRIVATE
  src/main.c
  src/tsr_model.c
  src/tsr_model_data.c
  src/tsr_image.c
)
target_include_directories(app PRIVATE src)

# Zephyr defaults the whole image to -Os. That costs the convolution loop its
# vectorisation: measured on x86 over the 364-image validation set, the same
# kernel runs 3.45 ms/image at -Os and 1.47 ms at -O3. Only the inference
# kernel is compiled for speed, so the flash cost is confined to one file
# instead of the whole image.
set_source_files_properties(
  src/tsr_model.c
  PROPERTIES COMPILE_OPTIONS "-O3"
)
