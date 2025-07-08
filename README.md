This is [opus](https://github.com/xiph/opus), packaged for Zig.
Unnecessary files have been deleted, and the build system has been replaced
with `build.zig`.

This is for Zig 0.14.1. The library will be updated only for STABLE Zig releases.

## Usage

### Adding to your project
1. Add this repository as a dependency in your `build.zig.zon`.
```sh
zig fetch --save https://github.com/rplstr/opus.git
```
2. In your `build.zig`, add the dependency and link it to your artifact.
```zig
// build.zig
const exe = b.addExecutable(...);

const opus = b.dependency("opus", .{
    .target = target,
    .optimize = optimize,
    // .fixed_point = true, // uncomment if you want fixed-point build
});

exe.linkLibrary(opus.artifact("opus"));

exe.addIncludePath(opus.path("include"));
```
3. Then just `@cImport` `opus/opus.h`.
```zig
const std = @import("std");

const c = @cImport({
    @cInclude("opus/opus.h");
});

pub fn main() !void {
    const version_string = c.opus_get_version_string();
    std.debug.print("opus {s}\n", .{version_string});
}
```
