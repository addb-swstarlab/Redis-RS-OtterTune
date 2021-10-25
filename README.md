# RS-OtterTune

Automatic Parameters Tuning System for Redis based on Non-linear machine learning methods.

![model_architecture](./img/rs-ottertune.jpg)

## Environments

- python: 3.7
- pytorch: 1.7.1
- redis: 5.0.2
- memtier_benchmark
- OS: Ubuntu 16.04.7 LTS (in GCP)
- CPU: Intel® Xeon® CPU @ 2.00GHz
- RAM: DIMM 16G

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

## Workloads

- \# of Requests: 1,000,000
- Key size of key-value data: 16 B
- Value size of key-value data: 128 B
- Read:Write Ratio
  - 1:0 (Write-Only)
  - 1:1

## Redis-Data-Generation

Since there is no available Redis workload dataset, it is required to carry out a step of generating data samples required for training.

https://github.com/addb-swstarlab/redis-sample-generation

## How to run?

To run RS-OtterTune just type:

```bash
$ python tuner/train.py --target <workload_num> --persistence <RDB or AOF> --rki <lasso, RF, XGB> 
```

By taking the best configuration file from the command above, execute the Redis server.

## Paper

Will update after acceptance journal

[Previous Study](https://www.eiric.or.kr/literature/ser_view.php?searchCate=literature&SnxGubun=INKO&mode=total&literature=Y&SnxGubun=INME&gu=INME000G2&cmd=qryview&SnxIndxNum=237774&q1_yy=2021&q1_mm=06&rownum=2&f1=MN&q1=Jieun%20Lee&totalCnt=21&kci=)

## Reference

```
@inproceedings{van2017automatic,
  title={Automatic database management system tuning through large-scale machine learning},
  author={Van Aken, Dana and Pavlo, Andrew and Gordon, Geoffrey J and Zhang, Bohan},
  booktitle={Proceedings of the 2017 ACM International Conference on Management of Data},
  pages={1009--1024},
  year={2017}
}
```