
#!/bin/bash

function PutFolder() {
	if [ $# -eq 2 ]; then
		HOST=arcc.uc.edu
		USER=mallorbc
		PORT=22
		SOURCE_FILE=$1
		DEST_FILE=$2
		COMMAND="put -r $SOURCE_FILE"
		sftp -P $PORT $USER@$HOST:$DEST_FILE <<<$COMMAND
	else
		HOST=$1
		USER=$2
		PORT=$3
		SOURCE_FILE=$4
		DEST_FILE=$5
		COMMAND="put -r $SOURCE_FILE"
		sftp -P $PORT $USER@$HOST:$DEST_FILE <<<$COMMAND
	fi

}