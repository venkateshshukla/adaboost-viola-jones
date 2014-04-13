%function for getting the data and the initial weights


function [ face_data,nonface_data ] = data_gen( A )

%data generation.
facefiles=dir('face/*.pgm');
nonfacefiles=dir('nonface/*.png');
%l=length(facefiles);
%m=length(nonfacefiles);

face_data=zeros(length(facefiles),3);
nonface_data=zeros(length(nonfacefiles),3);

%weights=[ones(l,1)./(2*l);ones(m,1)./(2*m)];
%weights=weights./(sum(weights));





for i=1:length(facefiles)
    
    
    filename=['face/',strcat(facefiles(i).name)];
    
    im=imread(filename);
    im=mat2gray(im);
    im=imresize(im,[24 24]);
    im=histeq(im);
    im=(im-mean(mean(im)))./(var(im(:)));
    
    %you must be thinking that why we are using this step. We told you that
    %we will be using integral images for that. okay, we will develop the
    %workaround. no worries!!
    im=imfilter(im,A,'same');
    %im=mat2gray(im);
    face_data(i,1)=mean(mean(im));
            
end
face_data(:,2)=ones(length(facefiles),1);
face_data(:,3)=ones(length(facefiles),1);

for i=1:length(nonfacefiles)
    
    
    filename=['nonface/',strcat(nonfacefiles(i).name)];
        
    im=imread(filename);
    im=mat2gray(im);
    im=imresize(im,[24 24]);
    im=histeq(im);
    im=(im-mean(mean(im)))./(var(im(:)));
    im=imfilter(im,A,'same');
    %im=mat2gray(im);
    nonface_data(i,1)=mean(mean(im));
    
        
end
nonface_data(:,2)=ones(length(nonface_data),1);
nonface_data(:,3)=-1.*ones(length(nonface_data),1);




end