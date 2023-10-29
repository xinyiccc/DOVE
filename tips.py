

#----------------------------Attention----------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)# --------------------------------
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

        
class MultiHeadAttention(nn.Module):  #positional encoding multihead attention
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        max_sequence_length=128
        self.max_sequence_length = max_sequence_length


        # Positional Encoding
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe = torch.zeros((max_sequence_length, emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # Linear layers
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
#---------------------------------Attention-----------------------------

# ----------------------------分阶段训练------------------------
# 一阶段训练和验证
        def train(self):
        
        img, label, test_data, test_label = self.get_source_data()

        #获得原始训练和测试numpy数据
        train_indices = random.sample(range(len(img)), 216)
        val_indices = random.sample(set(range(len(img))) - set(train_indices), 72)
        train_data_numpy = img[train_indices]
        train_label_numpy = label[train_indices]
        val_data_numpy = img[val_indices]
        val_label_numpy = label[val_indices]

        img= torch.from_numpy(img)
        label = torch.from_numpy(label)
        img = img.to(device).float()
        label = label.to(device).long()
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size1, shuffle=True)


        # 随机抽取200次实验做训练集
        
        train_data = img[train_indices]
        train_label = label[train_indices]
      
        # 随机抽取88次实验做验证集
        
        val_data = img[val_indices]
        val_label = label[val_indices]

        # 构建训练集和验证集的数据集和数据加载器
        train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size0, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size0, shuffle=True)
        

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        test_data = test_data.to(device).float()
        test_label = test_label.to(device).long()
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size1, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

      

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        

       # torch.save(self.model.module.state_dict(), 'model.pth')
        #averAcc = averAcc / num
       # print('The average accuracy is:', averAcc)
        
        print('The best accuracy is:', bestAcc)
       # self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
       
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
    
        return bestAcc,  Y_true, Y_pred #averAcc,
        writer.close()
        
        best_loss = float('inf')
        patience = 0
        for epoch in range(self.n_epochs_first):
            # 训练
            for i, (train_data, train_label) in enumerate(self.train_dataloader):
                # data augmentation
                aug_data0, aug_label0 = self.interaug(train_data_numpy, train_label_numpy,self.batch_size0)
              
                train_data = torch.cat((train_data, aug_data0))
                train_label = torch.cat((train_label, aug_label0))
                

                tok_train, outputs_train = self.model(train_data)
        

                train_loss = self.criterion_cls(outputs_train, train_label) 
                
                print('train_epoch:',epoch,'train_loss',train_loss)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for j, (val_data, vala_label) in enumerate(self.val_dataloader):
                    tok_evl,outputs_evl = self.model(val_data)
                    val_loss = self.criterion_cls(outputs_evl, vala_label)
                
                # 判断是否保存模型参数
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    print('best_val_loss',best_loss)
                else:
                    patience += 1
                    if patience == 300:
                        break
        
        #二阶段继续训练+测试
        best_loss_sec = float('inf')
        patience_sec = 0
        self.model.load_state_dict(torch.load('best_model.pth'))
        for e in range(self.n_epochs_sec):
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel,self.batch_size1)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))


                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label) 
                print('epoch_sec:',e,'loss:',loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                '''
                if loss < best_loss_sec:
                    best_loss_sec = loss
                    patience_sec = 0
                   
                else:
                    patience_sec += 1'''


            # out_epoch = time.time()


            # test process
            if loss < train_loss:
                self.model.eval()
                Tok, Cls = self.model(test_data)


                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                Y_true = test_label
                Y_pred = y_pred
                bestAcc = acc
                '''
                num = num + 1
                #averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred'''
                # stop training
                break
#--------------------------------分阶段训练-----------------------------