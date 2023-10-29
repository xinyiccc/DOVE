# 一阶段训练和验证
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