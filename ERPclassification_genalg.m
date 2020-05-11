%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% classification of HR-ASD infants with feature selection based on a genetic algorithm (G. Bussu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% setup

clear all;

erp_folder = ***;
data_folder=***;
cd(data_folder);

% execution flags

nrun=2; % change based on whether itś the first time you run the script or not (this will affect the partition)
n_validation=1;
gender_sensitivity=1;

do_nfeat=0;
do_repetitions=0;
do_freq=0;
do_compare=0;
do_compare_old=0;
do_plot=0;

classify_opt=1;
classify_freq=1;
classify_all=1;
classify_single=1;
classify_contrast=1;
classify_demo=0;
classify_static=1;
classify_shift=1;
classify_noise=1;
classify_freq_NF=0;
classify_freq_F=0;

check_nvt_prediction=0;
check_gender_prediction=0;


%% data

load('***')


% select only HR data
hrindx=find(data(:,2)>0);
data=data(hrindx,:);

% select only boys
if(gender_sensitivity==1)
maleindx=find(data(:,8)==0);
data=data(maleindx,:);
end



% hold-out 30% for later validation
if(nrun==1)
    partition = cvpartition(data(:,2),'HoldOut',0.3);
    if(gender_sensitivity==0)
        save('validation_partition.mat','partition');
    else
        save('validation_partition_boys.mat','partition');
    end
else
    if(gender_sensitivity==0)
        load('validation_partition.mat');
    else
         load('validation_partition_boys.mat');
    end
end

HR_dataset = data;
validation_data = data(partition.test,:);
training_data = data(partition.training,:);

% HR-ASD vs HR-no ASD
data = training_data;
I12 = union(find(data(:,2)==1),find(data(:,2)==2));
I3 = find(data(:,2)==3);

% set the data as [gender, contrasts, single ERP, age]
Contrasts_asd = [data(I3,9)-data(I3,10), data(I3,11)-data(I3,12),data(I3,13)-data(I3,14),data(I3,15)-data(I3,16),data(I3,17)-data(I3,18),data(I3,19)-data(I3,20),data(I3,21)-data(I3,22),data(I3,23)-data(I3,24),data(I3,25)-data(I3,26),data(I3,27)-data(I3,28),data(I3,29)-data(I3,30),data(I3,31)-data(I3,32),data(I3,33)-data(I3,39),data(I3,34)-data(I3,40),data(I3,35)-data(I3,41),data(I3,36)-data(I3,42),data(I3,37)-data(I3,43),data(I3,38)-data(I3,44)];
Contrasts_typatyp = [data(I12,9)-data(I12,10), data(I12,11)-data(I12,12),data(I12,13)-data(I12,14),data(I12,15)-data(I12,16),data(I12,17)-data(I12,18),data(I12,19)-data(I12,20),data(I12,21)-data(I12,22),data(I12,23)-data(I12,24),data(I12,25)-data(I12,26),data(I12,27)-data(I12,28),data(I12,29)-data(I12,30),data(I12,31)-data(I12,32),data(I12,33)-data(I12,39),data(I12,34)-data(I12,40),data(I12,35)-data(I12,41),data(I12,36)-data(I12,42),data(I12,37)-data(I12,43),data(I12,38)-data(I12,44)];
data_asd=[data(I3,8),Contrasts_asd];
data_typatyp=[data(I12,8),Contrasts_typatyp];
data_asd=[data_asd,data(I3,9:end),data(I3,5)];
data_typatyp=[data_typatyp,data(I12,9:end),data(I12,5)];

data_asd_training=data_asd;
data_typatyp_training=data_typatyp;

if gender_sensitivity==1
    data_asd=data_asd(:,2:end);
    data_typatyp=data_typatyp(:,2:end);
    
    data_asd_training=data_asd;
    data_typatyp_training=data_typatyp;
end

if nrun==1
    if gender_sensitivity==0
    save('training_sample.mat','data_asd_training','data_typatyp_training');
    else
        save('training_sample_boys.mat','data_asd_training','data_typatyp_training');
    end
end

%% folders for data analysis

mkdir erp_folder;
cd erp_folder;

if gender_sensitivity==1
    mkdir boys;
    cd boys;
end


%% test number of features
mkdir feature_number;
if do_nfeat==1
cd feature_number;


for ff=5:30
MC_EvolveTBLV2TSfixed(100,ff,200,data_typatyp,data_asd,[],['Results_asdVSall_f',num2str(ff)])
end

% plot to check
featn=[];
for ii=5:30
load(['Results_asdVSall_f',num2str(ii),'.mat'])
featn=[featn;Result(:,1)];
end
figure,plot(featn,'*')
end

%chosen n features
nfeat= 22;

%% repeated evolutionary process to explore the feature space
if do_repetitions==1
cd ..
mkdir repeated_evolution;
cd repeated_evolution;

for repetitions=56:100
MC_EvolveTBLV2TSfixed(100,nfeat,200,data_typatyp,data_asd,[],['Results_asdVSall_f',num2str(nfeat),'_r',num2str(repetitions)])
end

end


cd ../repeated_evolution
if do_freq==1
repauc=[];
for ii=1:100
load(['Results_asdVSall_f',num2str(nfeat),'_r',num2str(ii),'.mat'])
repauc=[repauc;Result];
end

figure,plot(repauc(:,1),'*');

% select top classifiers based on AUC>0.85
indx85=find(repauc(:,1)>0.85);
topauc=repauc(indx85,7:end);

% optimal feature set
Iopt=repauc(repauc(:,1)==max(repauc(:,1)),7:end);

%% frequency analysis on the top classifiers to identify the most relevant features for HR-ASD vs HR-noASD

freq=zeros(56,1);

for rr=1:size(topauc,1)
for cc=7:size(topauc,2)
freq(topauc(rr,cc),1)=freq(topauc(rr,cc),1)+1;
end
end
freq=freq/size(topauc,1);
bar(freq);

% highest incidence feature set
Ifreq=find(freq>0.8);

save('opt_freq_featlist.mat','Iopt','Ifreq');

else
    load('opt_freq_featlist.mat');
end
%% classification on the validation set
if gender_sensitivity==1
    folder= "***";
    cd(folder);
    load("opt_freq_featlist.mat");
    cd ../validation;
else 
cd ..
mkdir validation;
cd validation;
end

if n_validation==1
data = validation_data;
I12 = union(find(data(:,2)==1),find(data(:,2)==2));
I3 = find(data(:,2)==3);

% set the data as [gender, contrasts, single ERP]
Contrasts_asd = [data(I3,9)-data(I3,10), data(I3,11)-data(I3,12),data(I3,13)-data(I3,14),data(I3,15)-data(I3,16),data(I3,17)-data(I3,18),data(I3,19)-data(I3,20),data(I3,21)-data(I3,22),data(I3,23)-data(I3,24),data(I3,25)-data(I3,26),data(I3,27)-data(I3,28),data(I3,29)-data(I3,30),data(I3,31)-data(I3,32),data(I3,33)-data(I3,39),data(I3,34)-data(I3,40),data(I3,35)-data(I3,41),data(I3,36)-data(I3,42),data(I3,37)-data(I3,43),data(I3,38)-data(I3,44)];
Contrasts_typatyp = [data(I12,9)-data(I12,10), data(I12,11)-data(I12,12),data(I12,13)-data(I12,14),data(I12,15)-data(I12,16),data(I12,17)-data(I12,18),data(I12,19)-data(I12,20),data(I12,21)-data(I12,22),data(I12,23)-data(I12,24),data(I12,25)-data(I12,26),data(I12,27)-data(I12,28),data(I12,29)-data(I12,30),data(I12,31)-data(I12,32),data(I12,33)-data(I12,39),data(I12,34)-data(I12,40),data(I12,35)-data(I12,41),data(I12,36)-data(I12,42),data(I12,37)-data(I12,43),data(I12,38)-data(I12,44)];
data_asd=[data(I3,8),Contrasts_asd];
data_typatyp=[data(I12,8),Contrasts_typatyp];
if gender_sensitivity==1
    data_asd=Contrasts_asd;
    data_typatyp=Contrasts_typatyp;
end
data_asd=[data_asd,data(I3,9:end),data(I3,5)];
data_typatyp=[data_typatyp,data(I12,9:end),data(I12,5)];

save('validation_data.mat','data_asd','data_typatyp');
else 
        load('validation_data.mat')
        load('training_sample.mat')
    end
% optimal classifier
if classify_opt==1
mkdir optimal_classifier_val;
cd optimal_classifier_val;

final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,1,'optimal')

load('final_results_optimal_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,metrics,'optimal')
save('metrics_Iopt.mat','metrics');
end

if classify_freq==1
    cd ..
    mkdir freq_classifier_val;
cd freq_classifier_val;

final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ifreq,1,'frequent')

load('final_results_frequent_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ifreq,metrics,'frequent')
save('metrics_Ifreq.mat','metrics');
end

if gender_sensitivity==1
    Iall=1:1:55;
else
Iall=[1:1:56];
end

if classify_all==1
    cd ..
    mkdir all_classifier_val;
cd all_classifier_val;



final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iall,1,'all')

load('final_results_all_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iall,metrics,'all')
save('metrics_Iall.mat','metrics');
end

Ierp=[20:1:55];
if gender_sensitivity==1
    Ierp=Ierp-1;
end

if classify_single==1
    cd ..
    mkdir erp_classifier_val;
cd erp_classifier_val;



final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ierp,1,'erp')

load('final_results_erp_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ierp,metrics,'erp')
save('metrics_Ierp.mat','metrics');
end

Icontrast=[2:1:19];
if gender_sensitivity==1
    Icontrast=Icontrast-1;
end

if classify_contrast==1
    cd ..
    mkdir contrast_classifier_val;
cd contrast_classifier_val;



final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Icontrast,1,'contrast')

load('final_results_contrast_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Icontrast,metrics,'contrast')
save('metrics_Icontrast.mat','metrics');
end

Idemo=[1,56];

if classify_demo==1
    cd ..
    mkdir demo_classifier_val;
cd demo_classifier_val;



final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Idemo,1,'demo')

load('final_results_demo_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Idemo,metrics,'demo')
save('metrics_Idemo.mat','metrics');
end

Inoiseface=[14,17];

if classify_freq_NF==1
    cd ..
    mkdir noiseface_freq_classifier_val;
cd noiseface_freq_classifier_val;



final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Inoiseface,1,'noiseface')

load('final_results_noiseface_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Inoiseface,metrics,'noiseface')
save('metrics_Inoiseface.mat','metrics');
end

IstaticF=[33,37,40,41,52];

if classify_freq_F==1
    cd ..
    mkdir staticface_freq_classifier_val;
cd staticface_freq_classifier_val;



final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,IstaticF,1,'static')

load('final_results_static_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,IstaticF,metrics,'static')
save('metrics_Istatic.mat','metrics');
end

Inoise=[44:1:55];
if gender_sensitivity==1
    Inoise=Inoise-1;
end

if classify_noise==1
    cd ..
    mkdir noise_classifier_val;
cd noise_classifier_val;



final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Inoise,1,'noise')

load('final_results_noise_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Inoise,metrics,'noise')
save('metrics_Inoise.mat','metrics');
end

Ishift=[20:1:31];
if gender_sensitivity==1
    Ishift=Ishift-1;
end

if classify_shift==1
    cd ..
    mkdir shift_classifier_val;
cd shift_classifier_val;



final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ishift,1,'shift')

load('final_results_shift_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ishift,metrics,'shift')
save('metrics_Ishift.mat','metrics');
end

Istatic=[32:1:43];
if gender_sensitivity==1
    Istatic=Istatic-1;
end

if classify_static==1
    cd ..
    mkdir static_classifier_val;
cd static_classifier_val;



final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Istatic,1,'static')

load('final_results_static_validation10000boot1000rep.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Istatic,metrics,'static')
save('metrics_Istatic.mat','metrics');
end

if check_nvt_prediction
   
    cd ..
    mkdir nvt_prediction
    cd nvt_prediction
       
    [~,indx]=ismember(data(:,1),beh_data36(:,1));
    find(indx==0)
    databeh=beh_data36(indx(indx~=0),:);
    data=data(find(indx~=0),:);
    
    I12 = find(databeh(:,2)>=35);
    I3 = find(databeh(:,2)<35);


% set the data as [gender, contrasts, single ERP, age]
Contrasts_asd = [data(I3,9)-data(I3,10), data(I3,11)-data(I3,12),data(I3,13)-data(I3,14),data(I3,15)-data(I3,16),data(I3,17)-data(I3,18),data(I3,19)-data(I3,20),data(I3,21)-data(I3,22),data(I3,23)-data(I3,24),data(I3,25)-data(I3,26),data(I3,27)-data(I3,28),data(I3,29)-data(I3,30),data(I3,31)-data(I3,32),data(I3,33)-data(I3,39),data(I3,34)-data(I3,40),data(I3,35)-data(I3,41),data(I3,36)-data(I3,42),data(I3,37)-data(I3,43),data(I3,38)-data(I3,44)];
Contrasts_typatyp = [data(I12,9)-data(I12,10), data(I12,11)-data(I12,12),data(I12,13)-data(I12,14),data(I12,15)-data(I12,16),data(I12,17)-data(I12,18),data(I12,19)-data(I12,20),data(I12,21)-data(I12,22),data(I12,23)-data(I12,24),data(I12,25)-data(I12,26),data(I12,27)-data(I12,28),data(I12,29)-data(I12,30),data(I12,31)-data(I12,32),data(I12,33)-data(I12,39),data(I12,34)-data(I12,40),data(I12,35)-data(I12,41),data(I12,36)-data(I12,42),data(I12,37)-data(I12,43),data(I12,38)-data(I12,44)];
data_asd=[data(I3,8),Contrasts_asd];
data_typatyp=[data(I12,8),Contrasts_typatyp];
data_asd=[data_asd,data(I3,9:end),data(I3,5)];
data_typatyp=[data_typatyp,data(I12,9:end),data(I12,5)];

data_asd_training=data_asd;
data_typatyp_training=data_typatyp;

data = validation_data;
[~,indx]=ismember(data(:,1),beh_data36(:,1));
    find(indx==0)
    databeh=beh_data36(indx(indx~=0),:);
    data=data(find(indx~=0),:);
    
    I12 = find(databeh(:,2)>=35);
    I3 = find(databeh(:,2)<35);

% set the data as [gender, contrasts, single ERP]
Contrasts_asd = [data(I3,9)-data(I3,10), data(I3,11)-data(I3,12),data(I3,13)-data(I3,14),data(I3,15)-data(I3,16),data(I3,17)-data(I3,18),data(I3,19)-data(I3,20),data(I3,21)-data(I3,22),data(I3,23)-data(I3,24),data(I3,25)-data(I3,26),data(I3,27)-data(I3,28),data(I3,29)-data(I3,30),data(I3,31)-data(I3,32),data(I3,33)-data(I3,39),data(I3,34)-data(I3,40),data(I3,35)-data(I3,41),data(I3,36)-data(I3,42),data(I3,37)-data(I3,43),data(I3,38)-data(I3,44)];
Contrasts_typatyp = [data(I12,9)-data(I12,10), data(I12,11)-data(I12,12),data(I12,13)-data(I12,14),data(I12,15)-data(I12,16),data(I12,17)-data(I12,18),data(I12,19)-data(I12,20),data(I12,21)-data(I12,22),data(I12,23)-data(I12,24),data(I12,25)-data(I12,26),data(I12,27)-data(I12,28),data(I12,29)-data(I12,30),data(I12,31)-data(I12,32),data(I12,33)-data(I12,39),data(I12,34)-data(I12,40),data(I12,35)-data(I12,41),data(I12,36)-data(I12,42),data(I12,37)-data(I12,43),data(I12,38)-data(I12,44)];
data_asd=[data(I3,8),Contrasts_asd];
data_typatyp=[data(I12,8),Contrasts_typatyp];
data_asd=[data_asd,data(I3,9:end),data(I3,5)];
data_typatyp=[data_typatyp,data(I12,9:end),data(I12,5)];

final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,1,'optimal')

load('final_results_optimal.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,metrics,'optimal')
save('metrics_Iopt_nvt.mat','metrics');

final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ifreq,1,'frequent')

load('final_results_frequent.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ifreq,metrics,'frequent')
save('metrics_Ifreq_nvt.mat','metrics');
    
end

if check_gender_prediction
   
    cd ..
    mkdir gender_prediction
    cd gender_prediction
       
    
    I12 = find(data(:,8)==0);
    I3 = find(data(:,8)==1);


% set the data as [gender, contrasts, single ERP, age]
Contrasts_asd = [data(I3,9)-data(I3,10), data(I3,11)-data(I3,12),data(I3,13)-data(I3,14),data(I3,15)-data(I3,16),data(I3,17)-data(I3,18),data(I3,19)-data(I3,20),data(I3,21)-data(I3,22),data(I3,23)-data(I3,24),data(I3,25)-data(I3,26),data(I3,27)-data(I3,28),data(I3,29)-data(I3,30),data(I3,31)-data(I3,32),data(I3,33)-data(I3,39),data(I3,34)-data(I3,40),data(I3,35)-data(I3,41),data(I3,36)-data(I3,42),data(I3,37)-data(I3,43),data(I3,38)-data(I3,44)];
Contrasts_typatyp = [data(I12,9)-data(I12,10), data(I12,11)-data(I12,12),data(I12,13)-data(I12,14),data(I12,15)-data(I12,16),data(I12,17)-data(I12,18),data(I12,19)-data(I12,20),data(I12,21)-data(I12,22),data(I12,23)-data(I12,24),data(I12,25)-data(I12,26),data(I12,27)-data(I12,28),data(I12,29)-data(I12,30),data(I12,31)-data(I12,32),data(I12,33)-data(I12,39),data(I12,34)-data(I12,40),data(I12,35)-data(I12,41),data(I12,36)-data(I12,42),data(I12,37)-data(I12,43),data(I12,38)-data(I12,44)];
data_asd=[data(I3,8),Contrasts_asd];
data_typatyp=[data(I12,8),Contrasts_typatyp];
data_asd=[data_asd,data(I3,9:end),data(I3,5)];
data_typatyp=[data_typatyp,data(I12,9:end),data(I12,5)];

data_asd_training=data_asd;
data_typatyp_training=data_typatyp;

data = validation_data;
I12 = find(data(:,8)==0);
    I3 = find(data(:,8)==1);


% set the data as [gender, contrasts, single ERP]
Contrasts_asd = [data(I3,9)-data(I3,10), data(I3,11)-data(I3,12),data(I3,13)-data(I3,14),data(I3,15)-data(I3,16),data(I3,17)-data(I3,18),data(I3,19)-data(I3,20),data(I3,21)-data(I3,22),data(I3,23)-data(I3,24),data(I3,25)-data(I3,26),data(I3,27)-data(I3,28),data(I3,29)-data(I3,30),data(I3,31)-data(I3,32),data(I3,33)-data(I3,39),data(I3,34)-data(I3,40),data(I3,35)-data(I3,41),data(I3,36)-data(I3,42),data(I3,37)-data(I3,43),data(I3,38)-data(I3,44)];
Contrasts_typatyp = [data(I12,9)-data(I12,10), data(I12,11)-data(I12,12),data(I12,13)-data(I12,14),data(I12,15)-data(I12,16),data(I12,17)-data(I12,18),data(I12,19)-data(I12,20),data(I12,21)-data(I12,22),data(I12,23)-data(I12,24),data(I12,25)-data(I12,26),data(I12,27)-data(I12,28),data(I12,29)-data(I12,30),data(I12,31)-data(I12,32),data(I12,33)-data(I12,39),data(I12,34)-data(I12,40),data(I12,35)-data(I12,41),data(I12,36)-data(I12,42),data(I12,37)-data(I12,43),data(I12,38)-data(I12,44)];
data_asd=[data(I3,8),Contrasts_asd];
data_typatyp=[data(I12,8),Contrasts_typatyp];
data_asd=[data_asd,data(I3,9:end),data(I3,5)];
data_typatyp=[data_typatyp,data(I12,9:end),data(I12,5)];
    
    

Iopt=Iopt(2:end);

final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,1,'optimal')

load('final_results_optimal.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,metrics,'optimal')
save('metrics_Iopt_sex.mat','metrics');

final_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ifreq,1,'frequent')

load('final_results_frequent.mat')
metrics.AUC=AUC_tot;
metrics.acc=acc;
metrics.sens=sens_tot;
metrics.spec=spec_tot;
metrics.ppv=ppv_tot;
metrics.npv=npv_tot;
shuffle_classification_validationsample(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Ifreq,metrics,'frequent')
save('metrics_Ifreq_sex.mat','metrics');
    
end

%% compare classifiers
if do_compare==1
    
    cd ..
    cd validation/optimal_classifier_val
    
    load('metrics_Iopt.mat');
    metrics_opt=metrics;
    
    cd ../freq_classifier_val
    load('metrics_Ifreq.mat');
    metrics_frequent=metrics;
    
    cd ../all_classifier_val
    load('metrics_Iall.mat');
    metrics_all=metrics;
    
    cd ../contrast_classifier_val
    load('metrics_Icontrast.mat');
    metrics_contrast=metrics;
    
    cd ../erp_classifier_val
    load('metrics_Ierp.mat');
    metrics_single=metrics;
    
    cd ../demo_classifier_val
    load('metrics_Idemo.mat');
    metrics_demo=metrics;
    
    if gender_sensitivity~=1
    cd ../noiseface_freq_classifier_val
    load('metrics_Inoiseface.mat');
    metrics_noisefaceF=metrics;
    end
    
    if gender_sensitivity~=1
    cd ../staticface_freq_classifier_val
    load('metrics_Istatic.mat');
    metrics_staticF=metrics;
    end
    
    cd ../noise_classifier_val
    load('metrics_Inoise.mat');
    metrics_noise=metrics;
    
    cd ../shift_classifier_val
    load('metrics_Ishift.mat');
    metrics_shift=metrics;
    
    cd ../static_classifier_val
    load('metrics_Istatic.mat');
    metrics_static=metrics;
    
    if gender_sensitivity~=1
    cd ../nvt_prediction
    load('metrics_Iopt_nvt.mat');
    metrics_NVTopt=metrics;
    load('metrics_Ifreq_nvt.mat');
    metrics_NVTfreq=metrics;
    end
    
    if gender_sensitivity~=1
    cd ../gender_prediction
    load('metrics_Iopt_sex.mat');
    metrics_SEXopt=metrics;
    load('metrics_Ifreq_sex.mat');
    metrics_SEXfreq=metrics;
    end
    
    cd ..
    compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,Iall,metrics_opt,metrics_all,'optVSall')
compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,Idemo,metrics_opt,metrics_demo,'optVSdemo')
compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,Icontrast,metrics_opt,metrics_contrast,'optVScontrast')
compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,Ierp,metrics_opt,metrics_single,'optVSsingle')
compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,Inoise,metrics_opt,metrics_noise,'optVSnoise')
compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,Istatic,metrics_opt,metrics_static,'optVSstatic')
compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,Ishift,metrics_opt,metrics_shift,'optVSshift')
compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,Ifreq,metrics_opt,metrics_frequent,'optVSfrequent')
compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,Inoiseface,metrics_opt,metrics_noisefaceF,'optVSnoisefaceF')
compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Iopt,IstaticF,metrics_opt,metrics_staticF,'optVSstaticF')
% compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Icontrast,Iall,metrics_contrast,metrics_all,'contrastVSall')
% compare_classification(data_typatyp_training,data_asd_training,data_typatyp,data_asd,Icontrast,Ierp,metrics_contrast,metrics_single,'contrastVSsingle')

    
    
end

%% plot ROC-curve
%NB load data first
if do_plot==1
FontSize=20;

load('*/final_results_all.mat');
xr_avg=mean(xr,2);
yr_avg=mean(yr,2);
h1=plot(xr_avg,yr_avg,'LineWidth',4);
set(gcf,'PaperType','A4');
set(gcf,'PaperOrientation','portrait');
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 40 40]);
set(gca,'FontSize',FontSize);
set(gca,'FontName','Arial');
hold on;

load('*/final_results_contrast.mat');
xr_avg=mean(xr,2);
yr_avg=mean(yr,2);
plot(xr_avg,yr_avg,'r','LineWidth',4);
hold on

load('*/final_results_erp_NF+F.mat')
xr_avg=mean(xr,2);
yr_avg=mean(yr,2);
plot(xr_avg,yr_avg,'g','LineWidth',4);
hold on

load('*/final_results_erp_staticANDfacenoiseASD.mat');
xr_avg=mean(xr,2);
yr_avg=mean(yr,2);
plot(xr_avg,yr_avg,'k','LineWidth',4);
hold on

load('*/final_results_erp_highestfreqASD.mat');
xr_avg=mean(xr,2);
yr_avg=mean(yr,2);
plot(xr_avg,yr_avg,'m','LineWidth',4);
hold on

load('*/final_results_erp_optimalASD.mat');
xr_avg=mean(xr,2);
yr_avg=mean(yr,2);
plot(xr_avg,yr_avg,'c','LineWidth',4);
hold on

legend('Static Gaze','Dynamic gaze','Face vs Noise','Static Gaze + Face vs. Noise','Highest Frequency','Optimal','Location','NorthEastOutside')
title('ROC curve')
set(gca,'FontSize',FontSize)
set(gca,'FontName','Arial')
print('ROCs.png','-dpng','-r400');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot PC to visualize misclassifications
load('*/Documents/PhD/ERP_paper/validation/validation_data.mat')
load('*/Documents/PhD/ERP_paper/repeated_evolution/opt_freq_featlist.mat')

%% optiml classifier
load('*/Documents/PhD/ERP_paper/validation/optimal_classifier_val/final_results_optimal.mat')
check=[data_typatyp;data_asd];
check_opt=check(:,Iopt);

% PCA
check_opt_std=zscore(check_opt);
cov_data=cov(check_opt_std');
[V,D]=eig(cov_data);
[e,i] = sort(diag(D), 'descend');
V_sorted = V(:,i);

for cc=1:length(e)
explained = sum(e(1:cc))/sum(e);
if explained>0.85
break
end
end
PC_num = cc;

V_selected = V_sorted(:,1:PC_num);

% plot
FontSize=20;
set(gcf,'PaperType','A4');
set(gcf,'PaperOrientation','portrait');
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 40 40]);
set(gca,'FontSize',FontSize);
set(gca,'FontName','Arial');
hold on;
plot(V_selected(find(label==0),1),V_selected(find(label==0),2),'s','MarkerSize',10,'MarkerEdgeColor','blue','MarkerFaceColor',[.6 .6 1])
hold on;plot(V_selected(find(label==1),1),V_selected(find(label==1),2),'s','MarkerSize',10,'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6])
plot(V_selected(find(label~=predicted),1),V_selected(find(label~=predicted),2),'ko','MarkerSize',10,'LineWidth',4)

legend('HR-no ASD','HR-ASD','Misclassification','Location','SouthEast')
title('Classification plot: Optimal classifier')
xlabel('PC1')
ylabel('PC2')
set(gca,'FontSize',FontSize)
set(gca,'FontName','Arial')
print('Optimal_classification.png','-dpng','-r400');

%% most frequent classifier
load('*/Documents/PhD/ERP_paper/validation/freq_classifier_val/final_results_frequent.mat')
check=[data_typatyp;data_asd];
check_opt=check(:,Ifreq);

% PCA
check_opt_std=zscore(check_opt);
cov_data=cov(check_opt_std');
[V,D]=eig(cov_data);
[e,i] = sort(diag(D), 'descend');
V_sorted = V(:,i);

for cc=1:length(e)
explained = sum(e(1:cc))/sum(e);
if explained>0.85
break
end
end
PC_num = cc;

V_selected = V_sorted(:,1:PC_num);

% plot
figure,plot(V_selected(find(label==0),1),V_selected(find(label==0),2),'b*')
hold on;plot(V_selected(find(label==1),1),V_selected(find(label==1),2),'r*')
plot(V_selected(find(label~=predicted),1),V_selected(find(label~=predicted),2),'ko')

%% ERP contrast
Icontrast=[2:1:19];

load('*/Documents/PhD/ERP_paper/validation/contrast_classifier_val/final_results_contrast.mat')
check=[data_typatyp;data_asd];
check_opt=check(:,Icontrast);

% PCA
check_opt_std=zscore(check_opt);
cov_data=cov(check_opt_std');
[V,D]=eig(cov_data);
[e,i] = sort(diag(D), 'descend');
V_sorted = V(:,i);

for cc=1:length(e)
explained = sum(e(1:cc))/sum(e);
if explained>0.85
break
end
end
PC_num = cc;

V_selected = V_sorted(:,1:PC_num);

% plot
figure,plot(V_selected(find(label==0),1),V_selected(find(label==0),2),'b*')
hold on;plot(V_selected(find(label==1),1),V_selected(find(label==1),2),'r*')
plot(V_selected(find(label~=predicted),1),V_selected(find(label~=predicted),2),'ko')
