import cv2
from ultralytics import YOLO, SAM, NAS, RTDETR
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import warnings
warnings.filterwarnings("ignore")

names={
  0: 'ekskavator',
  1: 'tractor',
  2: 'truck',
  3: 'kran'
}

# ДБ
host = "mongodb+srv://questintext:admin123admin@cluster0.j3qnnqz.mongodb.net/?retryWrites=true&w=majority"

def save_data(id,data):
    client = MongoClient(host=host)
    db = client['testdb']
    data_collection = db['hakatoncollection']
    search_data=data_collection.find_one({"_id":id})
    if search_data=={}:
        try:
            inserted_data=data_collection.insert_one({"_id":id,'data':data})
            print("insert done")
        except:
            print('Ошибка сохранения данных')
            return False
    else:
        inserted_data=data_collection.replace_one({"_id":id},{'data':data})
        print("replace done")
        return True



########################
cap = cv2.VideoCapture('/hakatom/web/public/static/videos/test.mp4')# Путь до видео
########################
model = YOLO('/hakatom/model/model/last.pt') # Путь до модели
########################
objects = {0:[], 1:[], 2:[], 3:[]}
numer_of_frame = 0

doubts = 6*60

while True:
    ret, frame = cap.read()
    numer_of_frame += 1
    if not ret:
        break
    res = model.predict(frame)
    boxes = res[0].boxes.xywh.cpu().numpy()
    classes = res[0].boxes.cls.cpu().numpy()
    for ind, clas in enumerate(classes):
        clas = int(clas)
        if len(objects[clas]) == 0:
            # Открываем событие
            objects[clas] += [[0, boxes[ind][0], boxes[ind][1], doubts]] # Время которое может быть недоступен объект
            data = {'0':{'type':names[clas], 'start_work':str(numer_of_frame/6/3600)+':'+str(numer_of_frame/6/60)+':'+str(numer_of_frame/6)}}
            print(data)
            save_data(6461611, data)
        if numer_of_frame == 1:
            # Открываем событие
            objects[clas] += [[objects[clas][-1][0], boxes[ind][0], boxes[ind][1], 60*6]] # Время которое может быть недоступен объект
            data = {str(objects[clas][-1][0]):{'type':names[clas], 'start_work':str(numer_of_frame/6/3600)+
                                               ':'+str(numer_of_frame/6/60)+':'+str(numer_of_frame/6)}}
            print(data)
            save_data(6461611, data)
        else:
            min_dif = 1e3
            k = -1
            for num, i in enumerate(objects[clas].copy()):
                dif = np.sqrt((boxes[ind][0]-i[1])**2 + (boxes[ind][1]-i[2])**2)
                if min_dif > dif:
                    min_dif = dif
                    k = num
            if min_dif < 250:
                objects[clas][k] = [objects[clas][k][0], boxes[ind][0], boxes[ind][1], doubts]
            else:
                objects[clas] += [[objects[clas][-1][0]+1, boxes[ind][0], boxes[ind][1], doubts]]
                data = {str(objects[clas][-1][0]+1):{'type':names[clas], 'start_work':str(numer_of_frame/6//3600)+
                                ':'+str(numer_of_frame/6//60)+':'+str(numer_of_frame//6)+'.'+str(round(numer_of_frame*(1000/6), 3))}}
                print(numer_of_frame, min_dif, data)
                save_data(6461611, data)
            
    for index, i in enumerate(objects[clas]):
        i[-1] -= 1
        if i[-1] < 0:
            # Закрываем событие
            data = {str(i[0]):{'type':names[clas], 'end_work':str(numer_of_frame/6/3600)+':'+str(numer_of_frame/6/60)+':'+str(numer_of_frame/6)}}
            print(data)
            save_data(6461611, data)
            del objects[clas][index]
for index, i in enumerate(objects[clas]):
    # Закрываем событие
    data = {str(i[0]):{'type':names[clas], 'end_work':str(numer_of_frame/6/3600)+':'+str(numer_of_frame/6/60)+':'+str(numer_of_frame/6)}}
    print(data)
    save_data(6461611, data)
    print('Final save done')
    del objects[clas][index]