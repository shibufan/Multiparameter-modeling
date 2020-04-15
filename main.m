clear;
clc;

load('Training_Data.mat')
load('Real_Test_Data.mat') %read test data

delta_num_n=[0,-1,1];
delta_lr=[0,-0.01,0.01];

final_num_n=[17 17 6 6]; %Node number of the hidden layer
final_lr=[0.088910333106323 0.114720424756302 0.116821559089208 0.069624750145961];

num_n=final_num_n;
lr=final_lr;

training
testing

final_MAPE=MAPE;
final_MAPE1=MAPE1;
final_MAPE2=MAPE2;
final_MAPE3=MAPE3;
final_MAPE4=MAPE4;
% save('trained model/ANN/final/net','net1');
% save('trained model/ANN/final/net','net2','-append');
% save('trained model/ANN/final/net','net3','-append');
% save('trained model/ANN/final/net','net4','-append');
% save('trained model/ANN/final/net','ps_input_1','-append');
% save('trained model/ANN/final/net','ps_input_2','-append');
% save('trained model/ANN/final/net','ps_input_3','-append');
% save('trained model/ANN/final/net','ps_input_4','-append');
% save('trained model/ANN/final/net','ps_output_1','-append');
% save('trained model/ANN/final/net','ps_output_2','-append');
% save('trained model/ANN/final/net','ps_output_3','-append');
% save('trained model/ANN/final/net','ps_output_4','-append');

counter=1;
tt1=[1 1 1 1];
tt2=[1 1 1 1];
temperature=1000;
r=0.9999;

while final_MAPE>0.01
    if counter>10000
        break;
    end
    tt1(1)=mod(floor(rand()*10000),3)+1;
    tt1(2)=mod(floor(rand()*10000),3)+1;
    tt1(3)=mod(floor(rand()*10000),3)+1;
    tt1(4)=mod(floor(rand()*10000),3)+1;
    tt2(1)=mod(floor(rand()*10000),3)+1;
    tt2(2)=mod(floor(rand()*10000),3)+1;
    tt2(3)=mod(floor(rand()*10000),3)+1;
    tt2(4)=mod(floor(rand()*10000),3)+1;
    
    for i=1:4
        num_n(1)=final_num_n(1)+delta_num_n(tt1(1));
        num_n(2)=final_num_n(2)+delta_num_n(tt1(2));
        num_n(3)=final_num_n(3)+delta_num_n(tt1(3));
        num_n(4)=final_num_n(4)+delta_num_n(tt1(4));
        lr(1)=final_lr(1)+delta_lr(tt2(1))*rand();
        lr(2)=final_lr(2)+delta_lr(tt2(2))*rand();
        lr(3)=final_lr(3)+delta_lr(tt2(3))*rand();
        lr(4)=final_lr(4)+delta_lr(tt2(4))*rand();
    end
    if num_n(1)<=0
        num_n(1)=1;  
    end
    if num_n(2)<=0
        num_n(2)=1;  
    end
    if num_n(3)<=0
        num_n(3)=1;  
    end
    if num_n(4)<=0
        num_n(4)=1;  
    end
    %% Train ANN & SVM Model
    training

    %% Test model
    testing

    %%
    if MAPE1<final_MAPE1
        final_MAPE1=MAPE1;
        final_num_n(1)=num_n(1);
        final_lr(1)=lr(1);
        save('trained model/ANN/final/net','net1','-append');
        save('trained model/ANN/final/net','ps_input_1','-append');
        save('trained model/ANN/final/net','ps_output_1','-append');
    else
        if exp((final_MAPE1-MAPE1)/temperature)>rand()
            final_MAPE1=MAPE1;
            final_num_n(1)=num_n(1);
            final_lr(1)=lr(1);
            save('trained model/ANN/final/net','net1','-append');
            save('trained model/ANN/final/net','ps_input_1','-append');
            save('trained model/ANN/final/net','ps_output_1','-append');
        end
    end
    if MAPE2<final_MAPE2
        final_MAPE2=MAPE2;
        final_num_n(2)=num_n(2);
        final_lr(2)=lr(2);
        save('trained model/ANN/final/net','net2','-append');
        save('trained model/ANN/final/net','ps_input_2','-append');
        save('trained model/ANN/final/net','ps_output_2','-append');
    else
        if exp((final_MAPE2-MAPE2)/temperature)>rand()
            final_MAPE2=MAPE2;
            final_num_n(2)=num_n(2);
            final_lr(2)=lr(2);
            save('trained model/ANN/final/net','net2','-append');
            save('trained model/ANN/final/net','ps_input_2','-append');
            save('trained model/ANN/final/net','ps_output_2','-append');
        end
    end
    if MAPE3<final_MAPE3
        final_MAPE3=MAPE3;
        final_num_n(3)=num_n(3);
        final_lr(3)=lr(3);
        save('trained model/ANN/final/net','net3','-append');
        save('trained model/ANN/final/net','ps_input_3','-append');
        save('trained model/ANN/final/net','ps_output_3','-append');
    else
        if exp((final_MAPE3-MAPE3)/temperature)>rand()
            final_MAPE3=MAPE3;
            final_num_n(3)=num_n(3);
            final_lr(3)=lr(3);
            save('trained model/ANN/final/net','net3','-append');
            save('trained model/ANN/final/net','ps_input_3','-append');
            save('trained model/ANN/final/net','ps_output_3','-append');
        end
    end
    if MAPE4<final_MAPE4
        final_MAPE4=MAPE4;
        final_num_n(4)=num_n(4);
        final_lr(4)=lr(4);
        save('trained model/ANN/final/net','net4','-append');
        save('trained model/ANN/final/net','ps_input_4','-append');
        save('trained model/ANN/final/net','ps_output_4','-append');
    else
        if exp((final_MAPE4-MAPE4)/temperature)>rand()
            final_MAPE4=MAPE4;
            final_num_n(4)=num_n(4);
            final_lr(4)=lr(4);
            save('trained model/ANN/final/net','net4','-append');
            save('trained model/ANN/final/net','ps_input_4','-append');
            save('trained model/ANN/final/net','ps_output_4','-append');
        end
    end
    if MAPE<final_MAPE
        final_MAPE=MAPE;
    end
    counter=counter+1;
end