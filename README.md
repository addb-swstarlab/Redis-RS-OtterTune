# RS-OtterTune

Automatic Parameters Tuning System for Redis based on Non-linear machine learning methods.
![model_architecture](./img/rs-ottertune.jpg)

## Environments

- python: 3.7
- pytorch: 1.7.1
- redis: 5.0.2
- memtier_benchmark
- Ubuntu

## What is Redis?

Redis is often referred as a *data structures* server. What this means is that Redis provides access to mutable data structures via a set of commands, which are sent using a *server-client* model with TCP sockets and a simple protocol. So different processes can query and modify the same data structures in a shared way.

Data structures implemented into Redis have a few special properties:

- Redis cares to store them on disk, even if they are always served and modified into the server memory. This means that Redis is fast, but that is also non-volatile.
- Implementation of data structures stress on memory efficiency, so data structures inside Redis will likely use less memory compared to the same data structure modeled using an high level programming language.
- Redis offers a number of features that are natural to find in a database, like replication, tunable levels of durability, cluster, high availability.

Another good example is to think of Redis as a more complex version of memcached, where the operations are not just SETs and GETs, but operations to work with complex data types like Lists, Sets, ordered data structures, and so forth.

If you want to know more, this is a list of selected starting points:

- Introduction to Redis data types. http://redis.io/topics/data-types-intro
- Try Redis directly inside your browser. [http://try.redis.io](http://try.redis.io/)
- The full list of Redis commands. http://redis.io/commands
- There is much more inside the Redis official documentation. http://redis.io/documentation