%% Simulation of a balanced network 

clear

seed = 39;
rng(seed);

% Number of neurons in each population
N = 5000;
Ne=0.8*N;
Ni=0.2*N;

% Number of neurons in ffwd layer
Nx=0.2*N;

% Recurrent net connection probabilities
P=[0.1 0.1; 0.1 0.1];

% Ffwd connection probs
Px=[.1; .1];

% Mean connection strengths between each cell type pair
Jm=[25 -150; 112.5 -250]/sqrt(N);
Jxm=[180; 135]/sqrt(N);

% Time (in ms) for sim
T=50000;

% Time discretization
dt=.1; %ms

% Proportion of neurons in each population.
qe=Ne/N;
qi=Ni/N;
qf=Nx/N;

% Number of time bins
Nt=round(T/dt);
time=dt:dt:T;

% Extra stimulus: Istim is a time-dependent stimulus
% it is delivered to all neurons with weights given by JIstim.
% Specifically, the stimulus to neuron j at time index i is:
% Istim(i)*JIstim(j)
Istim=zeros(size(time)); 
Istim(time>0)=0;
jestim=0; 
jistim=0;
Jstim=sqrt(N)*[jestim*ones(Ne,1); jistim*ones(Ni,1)]; 

% Build mean field matrices
Q=[qe qi; qe qi];
Qf=[qf; qf];
W=P.*(Jm*sqrt(N)).*Q;
Wx=Px.*(Jxm*sqrt(N)).*Qf;

% Synaptic timescales
taux=10;
taue=8;
taui=4;

% J_NeNe = binornd(1,P(1,1),Ne,Ne);
% J_NeNi = binornd(1,P(1,1),Ne,Ni);
% J_NiNe = binornd(1,P(1,1),Ni,Ne);
% J_NiNi = binornd(1,P(1,1),Ni,Ni);
J_NeNx = binornd(1,P(1,1),Ne,Nx);
J_NiNx = binornd(1,P(1,1),Ni,Nx);

% Generate full connectivity matrices
tic
J=[Jm(1,1)*binornd(1,P(1,1),Ne,Ne) Jm(1,2)*binornd(1,P(1,1),Ne,Ni); ...
   Jm(2,1)*binornd(1,P(1,1),Ni,Ne) Jm(2,2)*binornd(1,P(1,1),Ni,Ni)];
Jx=[J_NeNx.*Jxm(1); J_NiNx.*Jxm(2)];

tGen=toc;
disp(sprintf('\nTime to generate connections: %.2f sec',tGen))


%%% Make (correlated) Poisson spike times for ffwd layer
%%% See Trousdale et al 2013 Front. in Comp Neur. for details on this
%%% algorithm
tic

% Correlation between the spike trains in the ffwd layer
% FFwd spike train rate (in kHz)
rx=10/1000;
c=0.2; % Max correlation
p_daughter_processes = c * rand(Nx,1); % samples in [0,c]
% p_daughter_processes = c * rand(Nx,1); % samples in [0,c]
% Timescale of correlation
taujitter=5;

if(c<1e-5) % If uncorrelated
    nspikeX=poissrnd(Nx*rx*T);
    st=rand(nspikeX,1)*T;
    sx=zeros(2,numel(st));
    sx(1,:)=sort(st);
    sx(2,:)=randi(Nx,1,numel(st)); % neuron indices
    clear st;
else % If correlated
    rm=rx/mean(p_daughter_processes); % Firing rate of mother process
    nstm=poissrnd(rm*T); % Number of mother spikes
    stm=rand(nstm,1)*T; % spike times of mother process    
    maxnsx=T*rx*Nx*1.2; % Max num spikes
    sx=zeros(2,maxnsx);
    ns=0;
    for j=1:Nx  % For each ffwd spike train
        ns0=binornd(nstm,p_daughter_processes(j)); % Number of spikes for this spike train
        st=randsample(stm,ns0); % Sample spike times randomly
        st=st+taujitter*randn(size(st)); % jitter spike times
        st=st(st>0 & st<T); % Get rid of out-of-bounds times
        ns0=numel(st); % Re-compute spike count
        sx(1,ns+1:ns+ns0)=st; % Set the spike times and indices        
        sx(2,ns+1:ns+ns0)=j;
        ns=ns+ns0;
    end

    % Get rid of padded zeros
    sx = sx(:,sx(1,:)>0);
    
    % Sort by spike time
    [~,I] = sort(sx(1,:));
    sx = sx(:,I);
    nspikeX=size(sx,2);
end
tGenx=toc;
disp(sprintf('\nTime to generate ffwd spikes: %.2f sec',tGenx))

% Neuron parameters
Cm=1;
gL=1/15;
EL=-72;
Vth=-50;
Vre=-75;
DeltaT=1;
VT=-55;

% Random initial voltages
V0=rand(N,1)*(VT-Vre)+Vre;

% Maximum number of spikes for all neurons
% in simulation. Make it 50Hz across all neurons
% If there are more spikes, the simulation will terminate
maxns=ceil(.05*N*T); % was 0.05.

% Indices of neurons to record currents, voltages
nrecord0=100; % Number to record from each population
Irecord=[randperm(Ne,nrecord0) randperm(Ni,nrecord0)+Ne];
numrecord=numel(Irecord); % total number to record

% Number of time bins to average over when recording
nBinsRecord=10;
dtRecord=nBinsRecord*dt;
timeRecord=dtRecord:dtRecord:T;
Ntrec=numel(timeRecord);

% Integer division function
IntDivide=@(n,k)(floor((n-1)/k)+1);

% Set initial voltage
V=V0;

% Preallocate memory
Ie=zeros(N,1);
Ii=zeros(N,1);
Ix=zeros(N,1);

IeRec=zeros(numrecord,Ntrec);
IiRec=zeros(numrecord,Ntrec);
IxRec=zeros(numrecord,Ntrec);
% VRec=zeros(numrecord,Ntrec);
% wRec=zeros(numrecord,Ntrec);

iFspike=1;
s=zeros(2,maxns);
nspike=0;
TooManySpikes=0;

tic
for i=1:numel(time)


    % Propogate ffwd spikes
    while(sx(1,iFspike)<=time(i) && iFspike<nspikeX)
        jpre=sx(2,iFspike);
        Ix=Ix+Jx(:,jpre)/taux;
        iFspike=iFspike+1;
    end
    
    
    % Euler update to V
    V=V+(dt/Cm)*(Istim(i)*Jstim+Ie+Ii+Ix+gL*(EL-V)+gL*DeltaT*exp((V-VT)/DeltaT));
    
    % Find which neurons spiked
    Ispike=find(V>=Vth);    
    
    % If there are spikes
    if(~isempty(Ispike))

        % Store spike times and neuron indices
        if(nspike+numel(Ispike)<=maxns)
            s(1,nspike+1:nspike+numel(Ispike))=time(i);
            s(2,nspike+1:nspike+numel(Ispike))=Ispike;
        else
            TooManySpikes=1;
            break;
        end
        
        % Update synaptic currents
        Ie=Ie+sum(J(:,Ispike(Ispike<=Ne)),2)/taue;    
        Ii=Ii+sum(J(:,Ispike(Ispike>Ne)),2)/taui;            
        
        % Update cumulative number of spikes
        nspike=nspike+numel(Ispike);
    end            
    
    % Euler update to synaptic currents
    Ie=Ie-dt*Ie/taue;
    Ii=Ii-dt*Ii/taui;
    Ix=Ix-dt*Ix/taux;
    
    % This makes plots of V(t) look better.
    % All action potentials reach Vth exactly. 
    % This has no real effect on the network sims
    V(Ispike)=Vth;
    
    % Store recorded variables
    ii=IntDivide(i,nBinsRecord); 
    IeRec(:,ii)=IeRec(:,ii)+Ie(Irecord);
    IiRec(:,ii)=IiRec(:,ii)+Ii(Irecord);
    IxRec(:,ii)=IxRec(:,ii)+Ix(Irecord);
%     VRec(:,ii)=VRec(:,ii)+V(Irecord);
    
    % Reset membrane potential
    V(Ispike)=Vre;
    
    
    if mod(i*dt,T/5) == 0 % print every x iterations.
        fprintf('At time %d...\n',i*dt/T);
    end
end
% Normalize recorded variables by # bins
IeRec=IeRec/nBinsRecord; 
IiRec=IiRec/nBinsRecord;
IxRec=IxRec/nBinsRecord;
% VRec=VRec/nBinsRecord;

% Get rid of padding in s
s=s(:,1:nspike); 
tSim=toc;
disp(sprintf('\nTime for simulation: %.2f min',tSim/60))

%% Analyze stats and dynamics of X
% sx(1,:) are the spike times
% sx(2,:) are the associated neuron indices
figure;
plot(sx(1,sx(2,:)<=Nx),sx(2,sx(2,:)<=Nx),'k.','Markersize',0.01)
xlabel('time (ms)')
ylabel('Neuron index')

% Time-dependent mean rates
dtRate=100; % ms
xRateT=hist(sx(1,sx(2,:)<=Nx),dtRate:dtRate:T)/(dtRate*Nx);

% Plot time-dependent rates
figure;hold on
plot((dtRate:dtRate:T)/1000,1000*xRateT, 'linewidth',3)
legend('r_x')
ylabel('rate (Hz)')
xlabel('time (s)')

% Compute spike count covariances and correlations

% Compute spike count covariances over windows of size
% winsize starting at time T1 and ending at time T2.
winsize=250; 
T1=T/2; % Burn-in period
T2=T;   % Compute covariances until end of simulation
tic
C=SpikeCountCov(sx,Nx,T1,T2,winsize);
toc

% Get mean spike count covariances over each sub-pop
[II,JJ]=meshgrid(1:Nx,1:Nx);

% Compute spike count correlations
% This takes a while, so make it optional
ComputeSpikeCountCorrs=1;
if(ComputeSpikeCountCorrs)
    % Get correlation matrix from cov matrix
    tic
    R=corrcov(C);
    toc
    dist_Rx = R(II<=Nx & JJ<II & isfinite(R));
    mRx=mean(R(II<=Nx & JJ<II & isfinite(R)))
end

figure; 
hist(dist_Rx)


%% Make a raster plot of first 500 neurons 
% s(1,:) are the spike times
% s(2,:) are the associated neuron indices
figure;
plot(s(1,s(2,:)<=Ne),s(2,s(2,:)<=Ne),'k.','Markersize',0.01)
xlabel('time (ms)')
ylabel('Neuron index')

%% Mean rate of each neuron (excluding burn-in period)
Tburn=T/2;
reSim=hist(s(2,s(1,:)>Tburn & s(2,:)<=Ne),1:Ne)/(T-Tburn);
riSim=hist(s(2,s(1,:)>Tburn & s(2,:)>Ne)-Ne,1:Ni)/(T-Tburn);

% Mean rate over E and I pops
reMean=mean(reSim);
riMean=mean(riSim);
disp(sprintf('\nMean E and I rates from sims: %.2fHz %.2fHz',1000*reMean,1000*riMean))

% Time-dependent mean rates
dtRate=100; % ms
eRateT=hist(s(1,s(2,:)<=Ne),1:dtRate:T)/(dtRate*Ne);
iRateT=hist(s(1,s(2,:)>Ne),1:dtRate:T)/(dtRate*Ni);

% Plot time-dependent rates
figure
plot((dtRate:dtRate:T)/1000,1000*eRateT, 'linewidth',3)
hold on
plot((dtRate:dtRate:T)/1000,1000*iRateT, 'linewidth',3)
legend('r_e','r_i')
ylabel('rate (Hz)')
xlabel('time (s)')


%% Compute spike count covariances and correlations

% Compute spike count covariances over windows of size
% winsize starting at time T1 and ending at time T2.
winsize=250; 
T1=T/2; % Burn-in period
T2=T;   % Compute covariances until end of simulation
tic
C=SpikeCountCov(s,N,T1,T2,winsize);
toc

% Get mean spike count covariances over each sub-pop
[II,JJ]=meshgrid(1:N,1:N);
mCee=mean(C(II<=Ne & JJ<=II));
mCei=mean(C(II<=Ne & JJ>Ne));
mCii=mean(C(II>Ne & JJ>II));

% Mean-field spike count cov matrix
mC=[mCee mCei; mCei mCii];

% Compute spike count correlations
% This takes a while, so make it optional
ComputeSpikeCountCorrs=1;
if(ComputeSpikeCountCorrs)
    
    % Get correlation matrix from cov matrix
    tic
    R=corrcov(C);
    toc
    
    mRee=mean(R(II<=Ne & JJ<=II & isfinite(R)));
    mRei=mean(R(II<=Ne & JJ>Ne & isfinite(R)));
    mRii=mean(R(II>Ne & JJ>II & isfinite(R)));

    % Mean-field spike count correlation matrix
    mR=[mRee mRei; mRei mRii]


end

figure; hist(R(II>Ne & JJ>II & isfinite(R)))

%% Compute CV of ISI, if desired.
% Column 1 has spike times and column 2 the neuron index
ComputeCV=0;
if ComputeCV~=0
    spikeTrain = transpose(s);
    % Sort neuron index in ascending order, keeping the correct spike time for
    % each one of them.
    spikeTrain = sortrows(spikeTrain,2);
    
    tic
    BadValue_index = zeros(N-1,1);
    spikes = cell(N,1);
    for i=1:N-1
        
        BadValue_index(i+1) = BadValue_index(i) + sum(spikeTrain(:,2) == i);
        
        spikes{i} = spikeTrain(BadValue_index(i)+1:BadValue_index(i+1),1);
        
    end
    for i=1:N
        if isempty(spikes{i})
            spikes{i} = [];
        end
    end
    ISI=cell(N,1);
    for i=1:N
        ISI{i} = diff(spikes{i}(:),1,1);
        sigma(i) = std(ISI{i}(:));
        mu(i) = mean(ISI{i}(:));
    end
    tSim=toc;
    disp(sprintf('\nTime for CV: %.2f min',tSim/60))
    
    CV_ISI = sigma./mu;
    CV_ISI = CV_ISI(~isnan(CV_ISI));
    
    % Plot distribution of ISI's.
    figure;
    histogram(CV_ISI(CV_ISI~=0),1000)
    
    MEAN_CV_ISI = mean(CV_ISI(CV_ISI~=0))
    
end

%% Check E, I, X currents for balance.
figure; hold on
plot(mean(IeRec))
plot(mean(IiRec))
plot(mean(IxRec))
plot(mean(IeRec)+mean(IiRec)+mean(IxRec))

IeRec1 = mean(IeRec);
IiRec2 = mean(IiRec);
IxRec3 = mean(IxRec);
