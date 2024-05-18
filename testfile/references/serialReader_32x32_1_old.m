% Code to read from 32 x 32 sensor array
% Subramanian Sundaram - subramanian.s88@gmail.com

function [s,flag] = serialReader_32x32_1(comPort)
close all
flag = 1;
delete(instrfindall);
s= serial(comPort);
set(s, 'DataBits', 8);
set(s, 'StopBits', 1);
set(s, 'InputBufferSize', 2050);
set(s, 'BaudRate', 150000);
set(s, 'Parity', 'none');
set(s, 'FlowControl', 'none');
set(s, 'Timeout', 1);
fopen(s);
a='b';
i=1;
st=[];

b=zeros(1024);
figure;
h=gcf;
set(h,'Position',[50 50 900 800]);

zdat=zeros(32,32);
%plotgraph=heatmap(zdat,'ColorLimits',[550 700],'Colormap', flipud(pink),'CellLabelColor', 'black');
plotgraph=imagesc(zdat,[10 150]);
colormap(flipud(gray));
axis image;
colorbar;
%plotgraph.GridVisible='off';
%plotgraph.CellLabelFormat='%0.0d';

pause(0.01);
res = [];
res2=[];
n=0

while (1)
    fprintf(s,'%c',17);
    pause(0.1);
    n=n+1;
    readval=fread(s, 2048, 'uint8');
    temp1=fread(s, 1, 'uint8');
    if(size(readval,1)==2048)
        readval_MSB=readval(1:2:end,:);
        readval_LSB=readval(2:2:end,:);
        readval=readval_MSB*32+readval_LSB;
        
        
        zdat=vec2mat(readval,32);
%         set(plotgraph, 'CData', zdat);
        res = cat(3, res, reshape(zdat, 32, 32, 1));
%         res2 = cat(1, res2, reshape(zdat, 1, 32, 32));
        if n==300
            cal=mean(res,3);
            z=1
        end
        
        if n>300
            
            zdat=minus(zdat,cal);
            set(plotgraph, 'CData', zdat);
        end

%         if n>120
%             break
%         end
    end
end
% save('01262019_01.mat', 'res');
res2 = reshape(res2,size(res2,1),[]);
xlswrite('test.xlsx',res2);
end
