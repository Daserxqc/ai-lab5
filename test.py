from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertConfig, BertModel

# Define the image trans forms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the dataset class
class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self,index):
        guid = self.data[index]['guid']
        image_path = './data/' + guid + '.jpg'
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.data[index]['label']
        return image, label

    def __len__(self):
        return len(self.data)

# Load the data
with open('./train.txt', 'r') as f:
    lines = f.readlines()

train_set = []
for line in lines[1:]:
    data = {}
    line = line.replace('\n','')
    guid, tag = line.split(',')
    if tag == 'positive':
        label = 0
    elif tag == 'neutral':
        label = 1
    else:
        label = 2
    data['guid'] = guid
    data['label'] = label
    train_set.append(data)

test_set = []
with open('./test_without_label.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        data = {}
        line = line.replace('\n','')
        guid, tag = line.split(',')
        # 检查标签是否正确
        if tag == 'null':
            label = -1
        elif tag not in ['positive', 'neutral', 'negative']:
            raise ValueError(f"Invalid tag '{tag}' in line '{line}'.")
        else:
            if tag == 'positive':
                label = 0
            elif tag == 'neutral':
                label = 1
            else:
                label = 2
        data['guid'] = guid
        data['label'] = label
        test_set.append(data)

def data_process(dataset):
  for data in dataset:
    guid = data['guid']
    image_path = './data/' + guid + '.jpg'
    image = Image.open(image_path).convert('RGB')
    
    array = np.array(image.resize((224, 224)))
    data['image'] = array.reshape((3, 224, 224))
    
    text_path = './data/' + guid + '.txt'
    f = open(text_path, 'r', errors='ignore')
    lines = f.readlines()
    # print(lines)
    text = ''
    for line in lines:
      text += line
    data['text'] = text

# print了一下共4000个，按照0.8：0.2的比例划分验证集
data_process(train_set)
train_set_num = 3200
valid_set_num = 800
train_set, valid_set = random_split(train_set, [train_set_num, valid_set_num])

data_process(test_set)

train_data = ImageDataset(train_set, transform=transform)
valid_data = ImageDataset(valid_set, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)

# Define the model
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=3, model_name='efficientnet-b0'):
        super(EfficientNetModel, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.model._fc.in_features, self.num_classes)

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier(x)
        return x

# Create the model and move it to the GPU if available
image_classifier = EfficientNetModel(num_classes=3, model_name='efficientnet-b0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_classifier.to(device)

# Define the optimizer and learning rate scheduler
optimizer = AdamW(image_classifier.parameters(), lr=1e-4, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*len(train_loader), num_training_steps=len(train_loader)*10)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
best_valid_acc = 0.0
for epoch in range(num_epochs):
    image_classifier.train()
    train_loss = 0.0
    for batch_num, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = image_classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    valid_loss = 0.0
    valid_correct = 0
    image_classifier.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = image_classifier(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            valid_correct += torch.sum(preds == labels.data)
    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    valid_acc = valid_correct.double() /valid_set_num
    print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}'.format(epoch+1, num_epochs, train_loss, valid_loss, valid_acc))

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载BERT预训练模型和tokenizer
model_path = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path)
bert_model = BertModel.from_pretrained(model_path, config=config)
bert_model.to(device)

class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

text_train = []
text_valid = []

for data in train_set:
    tokenized_text = tokenizer(data['text'], max_length=128, padding='max_length', truncation=True)
    tokenized_text['label'] = data['label']
    text_train.append(tokenized_text)

for data in valid_set:
    tokenized_text = tokenizer(data['text'], max_length=128, padding='max_length', truncation=True)
    tokenized_text['label'] = data['label']
    text_valid.append(tokenized_text)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index]['input_ids'], dtype=torch.long),
            torch.tensor(self.data[index]['attention_mask'], dtype=torch.long),
            torch.tensor(self.data[index]['label'], dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

train_loader = torch.utils.data.DataLoader(TextDataset(text_train), batch_size=25, shuffle=True)
valid_loader = torch.utils.data.DataLoader(TextDataset(text_valid), batch_size=25)

text_classifier = TextClassifier()
text_classifier.to(device)

epoch_num = 10
learning_rate = 1e-5
total_step = epoch_num * len(train_loader)

optimizer = torch.optim.Adam(text_classifier.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_step)

criterion = nn.CrossEntropyLoss()

for epoch in range(epoch_num):
    running_loss = 0
    for i, data in enumerate(train_loader):
        input_ids, attn_mask, labels = data
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        outputs = text_classifier(input_ids, attn_mask)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
    print('epoch: %d  loss: %.3f' % (epoch+1, running_loss/len(train_loader)))
    running_loss = 0

correct_num = 0
total_num = 0
with torch.no_grad():
    for data in valid_loader:
        input_ids, attn_mask, labels = data
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        outputs = text_classifier(input_ids, attn_mask)
        _, predicted = torch.max(outputs.data, 1)
        total_num += labels.size(0)
        correct_num += (predicted == labels).sum().item()

print('Validation Accuracy: %.3f%%' % (100 * correct_num / total_num))

# 融合模型

class MultimodalDataset(Dataset):
  def __init__(self, data):
    super(MultimodalDataset, self).__init__()
    self.data = data

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    guid = self.data[idx]['guid']
    input_ids = torch.tensor(self.data[idx]['input_ids'])
    attn_mask = torch.tensor(self.data[idx]['attn_mask'])
    image = torch.tensor(self.data[idx]['image'])
    label = self.data[idx].get('label')
    if label is None:
      label = -100
    label = torch.tensor(label)
    return guid, input_ids, attn_mask, image, label

def dataset_process(dataset):
  for data in dataset:
    tokenized_text = tokenizer(data['text'], max_length=128, padding='max_length', truncation=True)
    data['input_ids'] = tokenized_text['input_ids']
    data['attn_mask'] = tokenized_text['attention_mask']
    
dataset_process(train_set)
dataset_process(valid_set)
dataset_process(test_set)
     

train_loader = DataLoader(MultimodalDataset(train_set), batch_size=25, shuffle=True)
valid_loader = DataLoader(MultimodalDataset(valid_set), batch_size=25)
test_loader = DataLoader(MultimodalDataset(test_set), batch_size=25)

class MultimodalModel(nn.Module):
  def __init__(self, image_classifier, text_classifier, output_features, image_weight=0.5, text_weight=0.5):
    super(MultimodalModel, self).__init__()
    self.image_classifier = image_classifier
    self.text_classifier = text_classifier
    # 将最后的全连接层删除
    self.image_classifier.fc = nn.Sequential()  # (batch_num, 512)
    self.text_classifier.fc = nn.Sequential()    # (batch_num, 768)
    # 文本特征向量和图片特征向量的权重, 默认均为0.5
    self.image_weight = image_weight
    self.text_weight = text_weight
    self.fc1 = nn.Linear((3+768), output_features)
    self.fc2 = nn.Linear(output_features, 3)

  def forward(self, input_ids, attn_mask, image):
    image_output = self.image_classifier(image)
    text_output = self.text_classifier(input_ids, attn_mask)
    #print(image_output.shape)
    #print(text_output.shape)
    output = torch.cat([image_output, text_output], dim=1)
    #print(output.shape)
    output = self.fc1(output)
    output = self.fc2(output)
    return output 

multimodal_model = MultimodalModel(image_classifier=image_classifier, text_classifier=text_classifier, output_features=100, image_weight=0.5, text_weight=0.5)
multimodal_model.to(device)

epoch_num = 10
learning_rate = 1e-5
total_step = epoch_num * len(train_loader)

optimizer = AdamW(multimodal_model.parameters(), lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_step, num_training_steps=total_step)
criterion = nn.CrossEntropyLoss()

for epoch in range(epoch_num):
  running_loss = 0
  for i, data in enumerate(train_loader):
    _, input_ids, attn_mask, image, label = data
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    image = image.to(device)
    image = image.float()
    label = label.to(device)

    outputs = multimodal_model(input_ids=input_ids, attn_mask=attn_mask, image=image)
    # print(outputs.shape)
    loss = criterion(outputs, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    running_loss += loss.item()
  print('epoch: %d  loss: %.3f' % (epoch+1, running_loss/140))
  running_loss = 0
  
correct_num = 0
total_num = 0
with torch.no_grad():
  for data in valid_loader:
    _, input_ids, attn_mask, image, label = data
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    image = image.to(device)
    image = image.float()
    label = label.to(device)
    
    outputs = multimodal_model(input_ids=input_ids, attn_mask=attn_mask, image=image)
    _, predicted = torch.max(outputs.data, 1)
    for i in range(len(predicted.tolist())):
      total_num += label.size(0)
      correct_num += (predicted == label).sum().item()

print('Training Accuracy: %.3f%%' % (100 * correct_num / total_num))

test_dict = {}
with torch.no_grad():
  for data in test_loader:
    guid, input_ids, attn_mask, image, label = data
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    image = image.to(device)
    image = image.float()
    label = label.to(device)
    
    outputs = multimodal_model(input_ids=input_ids, attn_mask=attn_mask, image=image)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.tolist()
    for i in range(len(predicted)):
      id = guid[i]
      test_dict[id] = predicted[i]
      
with open('./test_without_label.txt', 'r') as f:
  lines = f.readlines()

f1 = open('./test.txt', 'w')
f1.write(lines[0])

for line in lines[1:]:
  # print(line)
  guid = line.split(',')[0]
  f1.write(guid)
  f1.write(',')
  label = test_dict[guid]
  if label == 0:
    f1.write('positive\n')
  elif label == 1:
    f1.write('neutral\n')
  else:
    f1.write('negative\n')