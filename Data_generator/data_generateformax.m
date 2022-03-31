clc
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Downlink
%Consider a square are of DxD m^2
%M distributed APs serves K terminals, they all randomly located in the area
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Inital parameters
M=15; %number of access points
K=5; %number of terminals

D=1; %in kilometer
tau=20;%training length
[U,S,V]=svd(randn(tau,tau));%U includes tau orthogonal sequences 

B=20; %Mhz
Hb = 15; % Base station height in m
Hm = 1.65; % Mobile height in m
f = 1900; % Frequency in MHz
aL = (1.1*log10(f)-0.7)*Hm-(1.56*log10(f)-0.8);
L = 46.3+33.9*log10(f)-13.82*log10(Hb)-aL;

power_f=0.2; %downlink power: 200 mW
noise_p = 10^((-203.975+10*log10(20*10^6)+9)/10); %noise power                                            
Pd = power_f/noise_p;%nomalized receive SNR
Pp=Pd;%pilot power

d0=0.01;%km
d1=0.05;%km

N=130000;
R_cf_min=zeros(1,N);%min rate, cell-free, without power allocation
R_cf_opt_min=zeros(1,N); %min rate, cell-free, with power allocation

R_sc_min=zeros(1,N);%small cell
R_sc_opt_min=zeros(1,N);

R_cf_user=zeros(N,K);
R_sc_user=zeros(N,K);

a1=zeros(K,N);
a2=zeros(K,N);
a3=zeros(K,N);
a4=zeros(K,N);

for n=1:N
n

%%%%%Randomly locations of M APs%%%%
AP=zeros(M,2,9);
AP(:,:,1)=unifrnd(-D/2,D/2,M,2);

%Wrapped around (8 neighbor cells)
D1=zeros(M,2);
D1(:,1)=D1(:,1)+ D*ones(M,1);
AP(:,:,2)=AP(:,:,1)+D1;

D2=zeros(M,2);
D2(:,2)=D2(:,2)+ D*ones(M,1);
AP(:,:,3)=AP(:,:,1)+D2;

D3=zeros(M,2);
D3(:,1)=D3(:,1)- D*ones(M,1);
AP(:,:,4)=AP(:,:,1)+D3;

D4=zeros(M,2);
D4(:,2)=D4(:,2)- D*ones(M,1);
AP(:,:,5)=AP(:,:,1)+D4;

D5=zeros(M,2);
D5(:,1)=D5(:,1)+ D*ones(M,1);
D5(:,2)=D5(:,2)- D*ones(M,1);
AP(:,:,6)=AP(:,:,1)+D5;

D6=zeros(M,2);
D6(:,1)=D6(:,1)- D*ones(M,1);
D6(:,2)=D6(:,2)+ D*ones(M,1);
AP(:,:,7)=AP(:,:,1)+D6;

D7=zeros(M,2);
D7=D7+ D*ones(M,2);
AP(:,:,8)=AP(:,:,1)+D7;

D8=zeros(M,2);
D8=D8- D*ones(M,2);
AP(:,:,9)=AP(:,:,1)+D8;

%Randomly locations of K terminals:
Ter=zeros(K,2,9);
Ter(:,:,1)=unifrnd(-D/2,D/2,K,2);

%Wrapped around (8 neighbor cells)
D1=zeros(K,2);
D1(:,1)=D1(:,1)+ D*ones(K,1);
Ter(:,:,2)=Ter(:,:,1)+D1;

D2=zeros(K,2);
D2(:,2)=D2(:,2)+ D*ones(K,1);
Ter(:,:,3)=Ter(:,:,1)+D2;

D3=zeros(K,2);
D3(:,1)=D3(:,1)- D*ones(K,1);
Ter(:,:,4)=Ter(:,:,1)+D3;

D4=zeros(K,2);
D4(:,2)=D4(:,2)- D*ones(K,1);
Ter(:,:,5)=Ter(:,:,1)+D4;

D5=zeros(K,2);
D5(:,1)=D5(:,1)+ D*ones(K,1);
D5(:,2)=D5(:,2)- D*ones(K,1);
Ter(:,:,6)=Ter(:,:,1)+D5;

D6=zeros(K,2);
D6(:,1)=D6(:,1)- D*ones(K,1);
D6(:,2)=D6(:,2)+ D*ones(K,1);
Ter(:,:,7)=Ter(:,:,1)+D6;

D7=zeros(K,2);
D7=D7+ D*ones(K,2);
Ter(:,:,8)=Ter(:,:,1)+D7;

D8=zeros(K,2);
D8=D8- D*ones(K,2);
Ter(:,:,9)=Ter(:,:,1)+D8;

sigma_shd=8; %in dB
D_cor=0.1;

%%%%%%Create the MxK correlated shadowing matrix %%%%%%%

    %%%%M correlated shadowing cofficients of M APs:
    Dist=zeros(M,M);%distance matrix
    Cor=zeros(M,M);%correlation matrix

    for m1=1:M
        for m2=1:M
            Dist(m1,m2) = min([norm(AP(m1,:,1)-AP(m2,:,1)), norm(AP(m1,:,1)-AP(m2,:,2)),norm(AP(m1,:,1)-AP(m2,:,3)),norm(AP(m1,:,1)-AP(m2,:,4)),norm(AP(m1,:,1)-AP(m2,:,5)),norm(AP(m1,:,1)-AP(m2,:,6)),norm(AP(m1,:,1)-AP(m2,:,7)),norm(AP(m1,:,1)-AP(m2,:,8)),norm(AP(m1,:,1)-AP(m2,:,9)) ]); %distance between AP m1 and AP m2
            Cor(m1,m2)=exp(-log(2)*Dist(m1,m2)/D_cor);
        end
    end
    A1 = chol(Cor,'lower');
    x1 = randn(M,1);
    sh_AP = A1*x1;
    for m=1:M
        sh_AP(m)=(1/sqrt(2))*sigma_shd*sh_AP(m)/norm(A1(m,:));
    end

    %%%%K correlated shadowing matrix of K terminal:
    Dist=zeros(K,K);%distance matrix
    Cor=zeros(K,K);%correlation matrix

    for k1=1:K
        for k2=1:K
            Dist(k1,k2)=min([norm(Ter(k1,:,1)-Ter(k2,:,1)), norm(Ter(k1,:,1)-Ter(k2,:,2)),norm(Ter(k1,:,1)-Ter(k2,:,3)),norm(Ter(k1,:,1)-Ter(k2,:,4)),norm(Ter(k1,:,1)-Ter(k2,:,5)),norm(Ter(k1,:,1)-Ter(k2,:,6)),norm(Ter(k1,:,1)-Ter(k2,:,7)),norm(Ter(k1,:,1)-Ter(k2,:,8)),norm(Ter(k1,:,1)-Ter(k2,:,9)) ]); %distance between Terminal k1 and Terminal k2
            Cor(k1,k2)=exp(-log(2)*Dist(k1,k2)/D_cor);
        end
    end
    A2 = chol(Cor,'lower');
    x2 = randn(K,1);
    sh_Ter = A2*x2;
    for k=1:K
        sh_Ter(k)=(1/sqrt(2))*sigma_shd*sh_Ter(k)/norm(A2(k,:));
    end

%%% The shadowing matrix:
Z_shd=zeros(M,K);
for m=1:M
    for k=1:K
        Z_shd(m,k)= sh_AP(m)+ sh_Ter(k);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create an MxK large-scale coefficients beta_mk
BETAA = zeros(M,K);
dist=zeros(M,K);
for m=1:M  
    for k=1:K
    [dist(m,k),index] = min([norm(AP(m,:,1)-Ter(k,:,1)), norm(AP(m,:,2)-Ter(k,:,1)),norm(AP(m,:,3)-Ter(k,:,1)),norm(AP(m,:,4)-Ter(k,:,1)),norm(AP(m,:,5)-Ter(k,:,1)),norm(AP(m,:,6)-Ter(k,:,1)),norm(AP(m,:,7)-Ter(k,:,1)),norm(AP(m,:,8)-Ter(k,:,1)),norm(AP(m,:,9)-Ter(k,:,1)) ]); %distance between Terminal k and AP m
    if dist(m,k)<d0
         betadB=-L - 35*log10(d1) + 20*log10(d1) - 20*log10(d0);
    elseif ((dist(m,k)>=d0) && (dist(m,k)<=d1))
         betadB= -L - 35*log10(d1) + 20*log10(d1) - 20*log10(dist(m,k));
    else
    betadB = -L - 35*log10(dist(m,k)) + Z_shd(m,k); %large-scale in dB
    end
    
    BETAA(m,k)=10^(betadB/10);                                   
    end

end

%% Pilot Asignment: (random choice)
Phii=zeros(tau,K);
for k=1:K
    Point=randi([1,tau]);
    Phii(:,k)=U(:,Point);
end

Phii_cf = Phii; % pilot set of cell-free systems
%% Create Gamma matrix (variances of the channel estimates)
Gammaa = zeros(M,K);
triul=zeros(M,K);
tridl=zeros(M,K);
%set the transceiver frequency responses
Abul=unifrnd(1-sqrt(0.03),1+sqrt(0.03),1,M);
Aful=unifrnd(1-sqrt(0.03),1+sqrt(0.03),1,K);

Ebul=exp(i.*unifrnd(-pi/3.4641,pi/3.4641,1,M));
Eful=exp(i.*unifrnd(-pi/3.4641,pi/3.4641,1,K));

Abdl=unifrnd(1-sqrt(0.03),1+sqrt(0.03),1,M);
Afdl=unifrnd(1-sqrt(0.03),1+sqrt(0.03),1,K);

Ebdl=exp(i.*unifrnd(-pi/3.4641,pi/3.4641,1,M));
Efdl=exp(i.*unifrnd(-pi/3.4641,pi/3.4641,1,K));

Bul=Abul.*Ebul;
Ful=Aful.*Eful;

Bdl=Abdl.*Ebdl;
Fdl=Afdl.*Efdl;

for m=1:M
    for k=1:K
        triul(m,k)=norm(Ful(k))^2*norm(Bul(m))^2*BETAA(m,k);
        tridl(m,k)=norm(Fdl(k))^2*norm(Bdl(m))^2*BETAA(m,k);
    end
end 

mau=zeros(M,K);
maure=zeros(M,K);
for m=1:M
    for k=1:K
        mau(m,k)=norm( (BETAA(m,:).^(1/2)).*(Phii_cf(:,k)'*Phii_cf))^2;
        maure(m,k)=norm( (triul(m,:).^(1/2)).*(Phii_cf(:,k)'*Phii_cf))^2;
    end
end

for m=1:M
    for k=1:K
         Gammaa(m,k)=tau*Pp*BETAA(m,k)^2/(tau*Pp*mau(m,k) + 1);
         Gammaare(m,k)=tau*Pp*triul(m,k)^2/(tau*Pp*maure(m,k) + 1);
    end
end

%% 1) Each AP has equal power allocations for K terminals
%%Compute etaa(m): (each AP transmits equal power to K terminals)
etaa=zeros(M,1);
etaare=zeros(M,1);
for m=1:M
    etaa(m)=1/(sum(Gammaa(m,:)));
    etaare(m)=1/(sum(Gammaare(m,:)));
end

%%Compute Rate
SINR=zeros(1,K);
R_cf=zeros(1,K);
R_cfre=zeros(1,K);

%Pilot contamination
PC = zeros(K,K);
PCre = zeros(K,K);
for ii=1:K
    for k=1:K
        PC(ii,k)=sum((etaa.^(1/2)).*((Gammaa(:,ii)./BETAA(:,ii)).*BETAA(:,k)))*Phii_cf(:,ii)'*Phii_cf(:,k);
        PCre(ii,k)=sum((etaare.^(1/2)).*((Gammaare(:,ii)./tridl(:,ii)).*tridl(:,k)))*Phii_cf(:,ii)'*Phii_cf(:,k);
    end
end
PC1=(abs(PC)).^2;
PC1re=(abs(PC)).^2;

for k=1:K
    num=0;
    numre=0;
    for m=1:M
        num=num + (etaa(m)^(1/2))*Gammaa(m,k);
        numre=numre + (etaare(m)^(1/2))*Gammaare(m,k)*Fdl(k)*Bdl(m)/(Ful(k)*Bul(m));
    end
    SINR(k) = Pd*num^2/(1 + Pd*sum(BETAA(:,k)) + Pd*sum(PC1(:,k)) - Pd*PC1(k,k) );
    SINRre(k) = Pd*norm(numre)^2/(1 + Pd*sum(tridl(:,k)) + Pd*sum(PC1re(:,k)) - Pd*PC1re(k,k));
    %Rate:
    R_cf(k) = log2(1+ SINR(k));
    R_cfre(k) = log2(1+ SINRre(k));
end


for k=1:K
a1(k,n)=R_cf(k);
end
R_cf_mean(n)=min(R_cf(:));


for k=1:K
a2(k,n)=R_cfre(k);
end

R_cf_meanre(n)=min(R_cfre(:));

tmin=2^R_cf_mean(n)-1;
tminre=2^R_cf_meanre(n)-1;
tmax=2^(2*R_cf_mean(n)+1.2)-1;
tmaxre=2^(2*R_cf_meanre(n)+1.2)-1;
epsi=max(tmin/5,0.01);
epsire=max(tminre/5,0.01);
PhiPhi=zeros(K,K);
for ii=1:K
   for k=1:K
       PhiPhi(ii,k)=norm(Phii_cf(:,ii)'*Phii_cf(:,k));
    end
end
BETAAn=BETAA*Pd;
tridln=tridl*Pd;
Gammaan=Gammaa*Pd;
Gammaanre=Gammaare*Pd;
X0=zeros(M,K);
y0=zeros(M,1);
Z0=zeros(K,K);


%cvx_solver sedumi
cvx_quiet true
            while( (tmax - tmin >= epsi)|as==0)

            tnext = (tmax+tmin)/2; 
           cvx_begin sdp
              variables X(M,K) y(M,1) Z(K,K)
              minimize(0)
              subject to
                for k=1:K
                     norm([sqrt(tnext)*[Z((1:(k-1)),k);Z(((k+1):K),k)].*[PhiPhi((1:(k-1)),k);PhiPhi(((k+1):K),k)]  ;  sqrt(tnext)*(BETAAn(:,k).^(1/2)).*y ; sqrt(tnext*Pd)]) <= (Gammaan(:,k))'*X(:,k) ;
                end
                for m=1:M
                    norm(((Gammaan(m,:)).^(1/2)).*X(m,:)) <= y(m); 
                    y(m)<=sqrt(Pd);
                end
                
                for k=1:K
                    for ii=1:K
                        sum( ((Gammaan(:,ii)./BETAA(:,ii)).*BETAA(:,k)).*X(:,ii)  ) <= Z(ii,k);
                    end
                end
                
                for m=1:M
                 for k=1:K
                    X(m,k)>=0;
                    
                 end
                end
               
            cvx_end


            % bisection
            if strfind(cvx_status,'Solved') % feasible
%            fprintf(1,'Problem is feasible ',tnext);
            tmin = tnext;
            as=1;
            X0=X;
            y0=y;
            Z0=Z;
            else % not feasible
 %           fprintf(1,'Problem not feasible ',tnext);
            tmax = tnext;
            as=0;
            end

            end



%Pilot contamination useful
PC = zeros(K,K);
Othernoise=zeros(K,K);
for ii=1:K
    for k=1:K
        PC(ii,k)=sum(X0(:,ii).*((Gammaa(:,ii)./BETAA(:,ii)).*BETAA(:,k)))*Phii_cf(:,ii)'*Phii_cf(:,k);
        Othernoise(ii,k)=sum((X0(:,ii).^2).*((Gammaa(:,ii).*BETAA(:,k))));
    end
end
PC1=(abs(PC)).^2;
Othernoise1=abs(Othernoise);

for k=1:K
    num=0;
    
    for m=1:M
        num=num + X0(m,k)*Gammaa(m,k);
        
    end
    SINR(k) = Pd*num^2/(1 + Pd*sum(Othernoise1(:,k)) + Pd*sum(PC1(:,k)) - Pd*PC1(k,k) );
    
    %Rate:
    
    a3(k,n)=log2(1+ SINR(k));
end
save(['C://Users//Y7000P//Desktop//research//myresearch//RL1_downlink_power//DDPG//data//datag',num2str(n),'.mat']);

end

R_cf_min=reshape(a1,1,N*K);
R_cf_minre=reshape(a2,1,N*K);
R_cf_opt_min =reshape(a3,1,N*K);
R_cf_opt_minre =reshape(a4,1,N*K);

Y=linspace(0,1,N*K);
hold on; box on;grid on;

% plot(0,0,'k','LineWidth',1.5);
% plot(0,0,'--k','LineWidth',1.5);
% plot(sort(R_cf_opt_min),Y(:),'k','LineWidth',1.5)
plot(sort(R_cf_min),Y(:),'k','LineWidth',1.5)
% plot(sort(R_cf_opt_minre),Y(:),'k','LineWidth',1.5)
% plot(sort(R_cf_minre),Y(:),'--k','LineWidth',1.5)
axis([-inf 3 -inf inf]);
set(0,'defaultfigurecolor','w');
xlabel(' Per-User SE [bit/s/Hz]','Interpreter','Latex');
ylabel('CDF','Interpreter','Latex');
legend({'without power control'},'Interpreter','Latex','Location','SouthEast');
set(gca,'Fontsize',14);



