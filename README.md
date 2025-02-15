# Auto License Plate Reader

## Overview
Auto License Plate Reader (ALPR) is a computer vision project that identifies license plates and recognizes their characters. This system is built using YOLOv8 for object detection and supports both webcam and video file inputs. The goal is to create an efficient and accurate license plate recognition system that can handle multiple plate formats.

## Features
- **License Plate Detection**: Detects license plates on vehicles.
- **Character Recognition**: Recognizes and extracts text from detected license plates.
- **Multiple Input Sources**: Supports real-time webcam input and video file processing.
- **SqlLite DB**: Stores real time numbers to a database
- **Live Notifications**: Send realtime notifications via telegram
  

## Installation
### Clone the Repository
```sh
git clone https://github.com/donsolo-khalifa/autoLicensePlateReader.git
cd autoLicensePlateReader
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Model Files
Add your license plate model to the `autoLicensePlateReader` directory and rename it to `best.pt`. You can download my pretrained model [here](https://drive.google.com/file/d/1xDEJnfR2ZASA4_wlRPk31KEyMUAHh7vn/view?usp=sharing).

### Video File
Ensure the video is present in the `autoLicensePlateReader` directory and name it `licensevid.mp4` or change the following line in `main.py` to switch to webcam mode:
```python
cap = cv2.VideoCapture("licensevid.mp4")  # Change to cap = cv2.VideoCapture(0) for webcam
```

## Usage
### Running the Detection
- Create an empty folder called `json` in the `autoLicensePlateReader`
- Run the following commands:
```sh
python sqlLiteDB.py
python main.py
```

### Telegram Bot
To enable live notifications:
1. Set up a Telegram bot using BotFather.
2. Uncomment the following lines in `main.py` (lines 122 and 123):
```python
message = "🚘 Detected License Plates:\n" + plate_text
threading.Thread(target=sendPlate, args=(message,)).start()
```
3. Create a `.env` file in the `autoLicensePlateReader` directory and add your token and chat ID:
```
TOKEN = add token here
CHAT_ID = add chat id here
```

## TODO
- Improve OCR accuracy for better character recognition.
- Enhance multi-format license plate support.
- Optimize detection speed for real-time performance.

## Contributing
Feel free to fork the repository and submit pull requests. Issues and suggestions are welcome!

## License
This project is open-source.

## Acknowledgments
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV Community
- PyTorch Developers
- [Sort tracking algorithm](https://github.com/abewley/sort)

