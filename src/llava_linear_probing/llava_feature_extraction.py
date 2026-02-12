class CXRDataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.df = df
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.df)
    
    def image_loader(self, image_name):
        image = io.imread(image_name)
        image = (((image - image.min()) / (image.max() - image.min()))*255).astype(np.uint8)
        image = np.stack((image, )*3)
        image = np.transpose(image, (1, 2, 0))
        image = self.augmentations(image)
        return image
    
    def __getitem__(self, index):
        y = self.df.at[self.df.index[index], 'mistral_binary']
        x = self.image_loader(self.df.at[self.df.index[index], 'image_paths'])
        y = torch.tensor([y], dtype=torch.float)
        return x, y, index
    
#Loading pre-divided training, validation, and testing data
train_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((336,336)),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4756, 0.4756, 0.4756], std=[0.3011, 0.3011, 0.3011])
                                    ])
other_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((336,336)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4756, 0.4756, 0.4756], std=[0.3011, 0.3011, 0.3011])
                                    ])

df_train = pd.read_csv('training_data.csv')
df_validation = pd.read_csv('validation_data.csv')
df_test = pd.read_csv('testing_data.csv')
datagen_train = CXRDataset(df =  df_train.copy(), augmentations = train_transform) 
datagen_val = CXRDataset(df = df_validation.copy(), augmentations = other_transform) 
datagen_test = CXRDataset(df = df_test.copy(),augmentations = other_transform) 

#Creating dataloaders
train_loader = DataLoader(dataset=datagen_train, shuffle=False, batch_size=32, num_workers=8)
val_loader = DataLoader(dataset=datagen_val,  shuffle=False, batch_size=32, num_workers=8)
test_loader = DataLoader(dataset=datagen_test,  shuffle=False, batch_size=32, num_workers=8)

#Loading model
model_id = "YuchengShi/llava-med-v1.5-mistral-7b-chest-xray"
full_model = LlavaForConditionalGeneration.from_pretrained(model_id)
vision_encoder = full_model.vision_tower
vision_encoder.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_encoder = vision_encoder.to(device)

#Feature Extraction
def extract_features(dataloader, df, model, save_path=None):
    features = []
    model.eval()
  
    with torch.no_grad():
        for images, _, idx in tqdm(dataloader, desc="Extracting Features", unit="batch"):
            images = images.to("cuda")
            outputs = model(images)[0]
            pooled_output = outputs.max(dim=1).values

            for i, index in tqdm(enumerate(idx), desc="Collecting Features", unit="index", leave=False):
                features.append((index.item(), pooled_output[i].cpu().numpy()))

    features.sort(key=lambda x: x[0])
    features = np.array([feature[1] for feature in features])
    df['extracted_features'] = list(features)

    if save_path:
        df.to_pickle(save_path)
        print(f"[INFO] Saved DataFrame with features to: {save_path}")

    return df

df_train_with_features = extract_features(train_loader,df_train,vision_encoder,save_path = "train_features.pkl")
print("Training set features extracted")

df_val_with_features = extract_features(val_loader,df_validation,vision_encoder,save_path = "val_features.pkl")
print("Validation set features extracted")

df_test_with_features = extract_features(test_loader,df_test,vision_encoder,save_path = "test_features.pkl")
print("Testing set features extracted")
