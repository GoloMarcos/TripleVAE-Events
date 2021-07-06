#download Datasets
[ ! -f "../Events.zip" ] && echo "Downloading Events.zip" && gdown --id 1SA8RmNwnN4zTDRRvKjkvXWaFrSpk22o0 -O "../Events.zip"

[ ! -d "../Events" ] && echo "Unziping Events.zip" && unzip ../Events.zip -d ../