clear;
clc;

%% 1.read data & models
load('Real_Test_Data.mat') %read test data

load('trained model/ANN/net.mat') %read ANN model
load('trained model/SVM/svm.mat') %read SVM model
%% 2.Testing
%1.SVM predict
% actual order of TF in test data
test_responses_size = size(real_test_responses(:,1),1);
for i=1:test_responses_size
    freq=real_test_responses{i,1}(:,1);
    data=real_test_responses{i,1}(:,2)+real_test_responses{i,1}(:,3)*1j;
    fit_data=rationalfit(freq,data);
    test_order_TF(i)=size(fit_data.A(:,1),1);
end

% normalize test data
SVM_Test_matrix = mapminmax('apply',real_test_candidates',PS);
SVM_Test_matrix = SVM_Test_matrix';
SVM_Test_label = test_order_TF';

% predict orders of TF
[predict_SVM_test_label,accuracy_SVM_test,prob_estimates2] = svmpredict(SVM_Test_label,SVM_Test_matrix,svm_model);
SVM_test_result = [SVM_Test_label predict_SVM_test_label];

figure(1);
plot(SVM_Test_label,'r *');
hold on;
plot(predict_SVM_test_label,'b o');
grid on;
legend('Actual orders','SVM outputs');
xlabel('Testing samples');
ylabel('Order');
axis([0 40 5.5 12.5]);
string = {'SVM Predict result(RBF Kernel function) for Test Data ';
          ['accuracy = ' num2str(accuracy_SVM_test(1)) '%']};
title(string);

% 2. ANN prediction
test_size=size(real_test_responses(:,1),1);
test_data=cell(test_size,1);
test_outputs=cell(test_size,1);

for i=1:test_size
    test_data{i,1}(:,1)=real_test_responses{i,1}(:,1);
    test_outputs{i,1}(:,1:2)=real_test_responses{i,1}(:,2:3);
    for j=1:size(test_data{i,1}(:,1),1)
        test_data{i,1}(j,2:4)=real_test_candidates(i,1:3);
    end
end

predict_outputs=cell(test_size,1);
for i=1:test_size
    P_test=test_data{i,1}';
    if SVM_Test_label(i,1)==6
        p_test = mapminmax('apply',P_test,ps_input_1);
        t_sim = sim(net1,p_test);
        T_sim = mapminmax('reverse',t_sim,ps_output_1);
    end
    
    if SVM_Test_label(i,1)==7
        p_test = mapminmax('apply',P_test,ps_input_2);
        t_sim = sim(net2,p_test);
        T_sim = mapminmax('reverse',t_sim,ps_output_2);
    end
    
    if SVM_Test_label(i,1)==8
        p_test = mapminmax('apply',P_test,ps_input_3);
        t_sim = sim(net3,p_test);
        T_sim = mapminmax('reverse',t_sim,ps_output_3);
    end
    
    if SVM_Test_label(i,1)==10
        p_test = mapminmax('apply',P_test,ps_input_4);
        t_sim = sim(net4,p_test);
        T_sim = mapminmax('reverse',t_sim,ps_output_4);
    end
    T_sim=T_sim';
    predict_outputs{i,1}=T_sim;
end

N=0;
PE=0;
APE=0;
error_matrix=cell(test_size,1);

PE1=0;
PE2=0;
PE3=0;
PE4=0;
APE1=0;
APE2=0;
APE3=0;
APE4=0;
N1=0;
N2=0;
N3=0;
N4=0;

for i=1:test_size
    PE_temp=0;
    APE_temp=0;
    N_temp=0;
    error_matrix{i,1}=abs((test_outputs{i,1}-predict_outputs{i,1})./test_outputs{i,1});
    complex_S11_test = test_outputs{i,1}(:,1)+test_outputs{i,1}(:,2)*1j;
    S11_test_abs=abs(complex_S11_test);
    complex_S11_predict = predict_outputs{i,1}(:,1)+predict_outputs{i,1}(:,2)*1j;
    S11_predict_abs=abs(complex_S11_predict);
    APE_matrix=abs((S11_test_abs-S11_predict_abs)./S11_test_abs);
    for j=1:size(test_outputs{i,1}(:,1),1)
        error_real=abs((test_outputs{i,1}(j,1)-predict_outputs{i,1}(j,1))/test_outputs{i,1}(j,1));
        error_imag=abs((test_outputs{i,1}(j,2)-predict_outputs{i,1}(j,2))/test_outputs{i,1}(j,2));
        PE_temp=PE_temp+sqrt(error_real^2+error_imag^2);
        APE_temp=APE_temp+APE_matrix(j);
        N_temp=N_temp+1;
    end
    PE=PE+PE_temp;
    APE=APE+APE_temp;
    N=N+N_temp;
    if predict_SVM_test_label(i,1)==6
        PE1=PE1+PE_temp;
        APE1=APE1+APE_temp;
        N1=N1+N_temp;
    end
    if predict_SVM_test_label(i,1)==7
        PE2=PE2+PE_temp;
        APE2=APE2+APE_temp;
        N2=N2+N_temp;
    end
    if predict_SVM_test_label(i,1)==8
        PE3=PE3+PE_temp;
        APE3=APE3+APE_temp;
        N3=N3+N_temp;
    end
    if predict_SVM_test_label(i,1)==10
        PE4=PE4+PE_temp;
        APE4=APE4+APE_temp;
        N4=N4+N_temp;
    end
end
MPE1=PE1/N1;
MAPE1=APE1/N1;
MPE2=PE2/N2;
MAPE2=APE2/N2;
MPE3=PE3/N3;
MAPE3=APE3/N3;
MPE4=PE4/N4;
MAPE4=APE4/N4;
MPE=PE/N;
MAPE=APE/N;

disp(['MPE1: ' num2str(MPE1*100) '%'])
disp(['MAPE1: ' num2str(MAPE1*100) '%'])
disp(['MPE2: ' num2str(MPE2*100) '%'])
disp(['MAPE2: ' num2str(MAPE2*100) '%'])
disp(['MPE3: ' num2str(MPE3*100) '%'])
disp(['MAPE3: ' num2str(MAPE3*100) '%'])
disp(['MPE4: ' num2str(MPE4*100) '%'])
disp(['MAPE4: ' num2str(MAPE4*100) '%'])
disp(['MPE: ' num2str(MPE*100) '%'])
disp(['MAPE: ' num2str(MAPE*100) '%'])

for i=1:test_size
    wf = real_test_responses{i,1}(:,1);
    complex_S11_test = test_outputs{i,1}(:,1)+test_outputs{i,1}(:,2)*1j;
    S11_test_abs=abs(complex_S11_test);
    S11_test_dB = 20*log10(S11_test_abs);
    complex_S11_predict = predict_outputs{i,1}(:,1)+predict_outputs{i,1}(:,2)*1j;
    S11_predict_abs=abs(complex_S11_predict);
    S11_predict_dB = 20*log10(S11_predict_abs);
    figure(i+1);
    plot(wf,S11_test_dB,'r','LineWidth', 0.5);
    hold on;
    plot(wf,S11_predict_dB,'g','LineWidth', 0.5);
    legend('Actual responses','Predict responses')
    xlabel('Freq. in GHz')
    ylabel ('S_1_1 in dB')
    string = {['Multiparameter modeling predict result for testing sample ' num2str(i)];
        ['MAPE = ' num2str(MAPE*100) '%']};
    title(string);
    grid on
end