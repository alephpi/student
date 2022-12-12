# guidelines

## run zookeeper

Linux:
`bin/zookeeper-server-start.sh config/zookeeper.properties`

Win:
`.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties`

## run server

Linux:
`bin/kafka-server-start.sh config/server.properties`

Win:
`.\bin\windows\kafka-server-start.bat .\config\server.properties`

## create topic

`bin/kafka-topics.sh --create --topic topic_name --bootstrap-server localhost:9092`

## check creation

`bin/kafka-topics.sh --describe --topic topic_name --bootstrap-server localhost:9092`

## delete topic

`bin/kafka-topics.sh --delete --topic topic_name --bootstrap-server localhost:9092`

## check deletion

`bin/kafka-topics.sh --describe --topic topic_name --bootstrap-server localhost:9092`

for windows: https://www.cnblogs.com/asd14828/p/13529487.html
