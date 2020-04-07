clear;
clc;
%% ANN parameters
num_n=[300 300 250 200]; %Node number of the hidden layer
num_epochs=[50 50 100 100]; %Number of epochs
goal=[1e-8 1e-8 1e-8 1e-8]; %Optimization goal
lr=[0.1 0.1 0.05 0.05]; %Learning rate
%% 1.read data
load Training_Data.mat

%% 2.Vector fitting of S-parameters
responses_size = size(responses(:,1),1);
for i=1:responses_size
    freq=responses{i,1}(:,1);
    data=responses{i,1}(:,2)+responses{i,1}(:,3)*1j;
    fit_data=rationalfit(freq,data);
    order_TF(i)=size(fit_data.A(:,1),1);
end

%% 3.classification
Max_order_TF=max(order_TF);
Min_order_TF=min(order_TF);
categories_order_TF=categorical(order_TF);
Categories_order_TF=categories(categories_order_TF);
num_categories=size(Categories_order_TF,1);
candidates_categories=cell(num_categories,1);
responses_categories=cell(num_categories,1);
for i=1:num_categories
    k=1;
    for j=1:size(order_TF,2)
        if order_TF(j)==str2double(Categories_order_TF{i,1})
            candidates_categories{i,1}(k,:)=candidates(j,:);
            responses_categories{i,1}{k,1}=responses{j,1};
            k=k+1;
        end
    end
end

%% 4.ANN training
%% 4.1 Category 1 (order = 6)
temp_train_1=responses_categories{1,1};
temp_candidates_1=candidates_categories{1,1};
for i=1:size(temp_train_1,1)
    for j=1:size(temp_train_1{i,1},1)
        temp_train_1{i,1}(j,4:6)=temp_candidates_1(i,1:3);
    end
end
temp_train_cat1=[];
for i=1:size(temp_train_1,1)
    temp_train_cat1=cat(1,temp_train_cat1,temp_train_1{i,1});
end
P_train_1 = temp_train_cat1(:,[1 4:6])';
T_train_1 = temp_train_cat1(:,[2 3])';

% normalization
[p_train_1, ps_input_1] = mapminmax(P_train_1,0,1);

[t_train_1, ps_output_1] = mapminmax(T_train_1,-1,1);

% create net1
net1=newff(p_train_1,t_train_1,num_n(1));
 
% set training parameters 
net1.trainParam.epochs = num_epochs(1);
net1.trainParam.goal = goal(1);
net1.trainParam.lr = lr(1);

net1 = train(net1,p_train_1,t_train_1);

%% 4.2 Category 2 (order = 7)
temp_train_2=responses_categories{2,1};
temp_candidates_2=candidates_categories{2,1};
for i=1:size(temp_train_2,1)
    for j=1:size(temp_train_2{i,1},1)
        temp_train_2{i,1}(j,4:6)=temp_candidates_2(i,1:3);
    end
end
temp_train_cat2=[];
for i=1:size(temp_train_2,1)
    temp_train_cat2=cat(1,temp_train_cat2,temp_train_2{i,1});
end
P_train_2 = temp_train_cat2(:,[1 4:6])';
T_train_2 = temp_train_cat2(:,[2 3])';

% normalization
[p_train_2, ps_input_2] = mapminmax(P_train_2,0,1);

[t_train_2, ps_output_2] = mapminmax(T_train_2,-1,1);

% create net2
net2=newff(p_train_2,t_train_2,num_n(2));
 
% set training parameters 
net2.trainParam.epochs = num_epochs(2);
net2.trainParam.goal = goal(2);
net2.trainParam.lr = lr(2);

net2 = train(net2,p_train_2,t_train_2);

%% 4.3 Category 3 (order = 8)
temp_train_3=responses_categories{3,1};
temp_candidates_3=candidates_categories{3,1};
for i=1:size(temp_train_3,1)
    for j=1:size(temp_train_3{i,1},1)
        temp_train_3{i,1}(j,4:6)=temp_candidates_3(i,1:3);
    end
end
temp_train_cat3=[];
for i=1:size(temp_train_3,1)
    temp_train_cat3=cat(1,temp_train_cat3,temp_train_3{i,1});
end
P_train_3 = temp_train_cat3(:,[1 4:6])';
T_train_3 = temp_train_cat3(:,[2 3])';

% normalization
[p_train_3, ps_input_3] = mapminmax(P_train_3,0,1);

[t_train_3, ps_output_3] = mapminmax(T_train_3,-1,1);

% create net3
net3=newff(p_train_3,t_train_3,num_n(3));
 
% set training parameters 
net3.trainParam.epochs = num_epochs(3);
net3.trainParam.goal = goal(3);
net3.trainParam.lr = lr(3);

net3 = train(net3,p_train_3,t_train_3);

%% 4.4 Category 4 (order = 10)
temp_train_4=responses_categories{4,1};
temp_candidates_4=candidates_categories{4,1};
for i=1:size(temp_train_4,1)
    for j=1:size(temp_train_4{i,1},1)
        temp_train_4{i,1}(j,4:6)=temp_candidates_4(i,1:3);
    end
end
temp_train_cat4=[];
for i=1:size(temp_train_4,1)
    temp_train_cat4=cat(1,temp_train_cat4,temp_train_4{i,1});
end
P_train_4 = temp_train_cat4(:,[1 4:6])';
T_train_4 = temp_train_cat4(:,[2 3])';

% normalization
[p_train_4, ps_input_4] = mapminmax(P_train_4,0,1);

[t_train_4, ps_output_4] = mapminmax(T_train_4,-1,1);

% create net4
net4=newff(p_train_4,t_train_4,num_n(4));
 
% set training parameters 
net4.trainParam.epochs = num_epochs(4);
net4.trainParam.goal = goal(4);
net4.trainParam.lr = lr(4);

net4 = train(net4,p_train_4,t_train_4);

%% save ANN model
save('net','net1');
save('net','net2','-append');
save('net','net3','-append');
save('net','net4','-append');
save('net','ps_input_1','-append');
save('net','ps_input_2','-append');
save('net','ps_input_3','-append');
save('net','ps_input_4','-append');
save('net','ps_output_1','-append');
save('net','ps_output_2','-append');
save('net','ps_output_3','-append');
save('net','ps_output_4','-append');

%% 5.SVM training based on libsvm
% normalization
[SVM_Train_matrix,PS] = mapminmax(candidates');
SVM_Train_matrix = SVM_Train_matrix';

SVM_Train_label = order_TF';

% find optimal c/g parameters----cross validation method
% [bestacc,bestc,bestg] = SVM_Search_cg(SVM_Train_label,SVM_Train_matrix,-50,50,-50,50,4,1,1);
% cmd=sprintf('-t 2 -c %f -g %f ',bestc,bestg);

% use optimal parameters to create & train svm model
cmd=sprintf('-t 2 -c %f -g %f ',2^26,2^-28);
svm_model = svmtrain(SVM_Train_label,SVM_Train_matrix,cmd);

% predict
[predict_SVM_train_label,accuracy_SVM_train,prob_estimates1] = svmpredict(SVM_Train_label,SVM_Train_matrix,svm_model);
SVM_train_result = [SVM_Train_label predict_SVM_train_label];

%% save SVM model
save('svm','svm_model');
save('svm','PS','-append');
