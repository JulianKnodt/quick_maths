# Quick Maths

A basic package for vector and matrix arithmetic which uses rust's unstable const generic
feature. I just need something that does math conveniently and don't want to have to deal with
NDArray or the like...

## Some things that make usage smoother

#### Default types

Unless you explicitly need to change it, the type is set to a reasonable default(f32). If you
want to change the type of the default, this can be changed by setting a compilation flag. This
allows for easy usage of the library without worrying unless you need to.
