% TPOSE budgets oxygen
% Ariane Verdy, Sept 2021

clear all
close all

cd /data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/diags/
addpath ~/scripts_m

% oxygen 5-day averages

tbeg = datenum(2012,1,1);
tend = datenum(2013,1,0);

time = datenum(2004,1,3):5:datenum(2019,1,3);

t1=find(tbeg<=time,1,'first');
tmax=find(tend>=time,1,'last');

time = time(t1:tmax);
nt = length(time);

% 5 day
dts = 240;
ts = dts:dts:400000;
ts = ts(t1:tmax);

load /home/averdy/tpose/grid_6/grid XC YC RC Depth hFacC
[nx,ny,nz] = size(hFacC);
x1=1;x2=1127;y1=1;y2=335;
lon = XC(x1:x2,1);
lat = YC(1,y1:y2);
depth = RC;



% select area
x = x1:x2; 
y = y1:y2; 
z = 1:50; 

% model grid
load ~/tpose/grid_6/grid.mat
XC = XC(x,y);
YC = YC(x,y);
RC = RC(z);
hFacC = hFacC(x,y,z);
[nx,ny,nz] = size(hFacC);
dz = permute(repmat(DRF(z),[1,nx,ny]),[2,3,1]).*hFacC;
dzMat = permute(repmat(DRF(z),[1,nx,ny]),[2,3,1]).*hFacC;
dzMatF = permute(repmat(DRF(z),[1,nx,ny]),[2,3,1]);

% cell volume, face areas (for flux calculations)
volume = zeros(nx,ny,nz);
areaWest = zeros(nx+1,ny,nz);
areaSouth = zeros(nx,ny+1,nz);
areaTop = zeros(nx,ny,nz+1);
for k=1:nz
 volume(:,:,k) = hFacC(:,:,k).*RAC(x,y)*DRF(k);
 areaTop(:,:,k) = RAC(x,y);
 if x(end)==1128
  areaWest(:,:,k)  = DYG([x 1],y).*DRF(k).*hFacW([x 1],y,k);
  areaWest_noh(:,:,k)  = DYG([x 1],y).*DRF(k);
 else
  areaWest(:,:,k)  = DYG([x x(end)+1],y).*DRF(k).*hFacW([x x(end)+1],y,k);
  areaWest_noh(:,:,k)  = DYG([x x(end)+1],y).*DRF(k);
 end
 if y(end)==336
  areaSouth(:,:,k) = DXG(x,y).*DRF(k).*hFacS(x,y,k);
  areaSouth(:,end+1,k) = NaN*areaSouth(:,end,k);
  areaSouth_noh(:,:,k) = DXG(x,y).*DRF(k);
  areaSouth_noh(:,end+1,k) = NaN*areaSouth_noh(:,end,k);
 else
  areaSouth(:,:,k) = DXG(x,[y y(end)+1]).*DRF(k).*hFacS(x,[y y(end)+1],k);
  areaSouth_noh(:,:,k) = DXG(x,[y y(end)+1]).*DRF(k);
 end
end
areaTop(:,:,nz+1) = RAC(x,y);
area = RAC(x,y);
RACMat = repmat(RAC(x,y),[1 1 nz]);
VVV=RACMat.*dzMat; 
Depth = Depth(x,y);
nLevels=numel(RC);

rhoconst = 1035;




% read diagnostics
% calculate tendencies in mol/m3/s
for t=1:nt

display(t)

% VOLUME BUDGET

% read SSH snapshots
tmp = rdmds('diag_eta_snaps',[ts(t)-dts ts(t)],'rec',1);
ETAN_SNAP1 = tmp(x,y,1);
ETAN_SNAP2 = tmp(x,y,2);
% calculate tendency
dt = 86400*5;
tendV = repmat(1./Depth.*(ETAN_SNAP2−ETAN_SNAP1)/dt,[1 1 nz]);

% read freshwater flux
tmp = rdmds('diag_surf',ts(t),'rec',6);
oceFWflx = tmp(x,y);
% calculate surface forcing
forcV = repmat(oceFWflx,[1 1 nz])./(dzMat*rhoconst);
forcV(:,:,2:end) = 0;

% read velocities
vel = rdmds('diag_state_mass',ts(t),'rec',1:3);
UVELMASS = vel([x x(end)+1],y,z,1);
VVELMASS = vel(x,[y y(end)+1],z,2);
WVELMASS = vel(x,y,[z z(end)+1],3);
% calculate horizontal transport
% (U(V)VELMASS are already weighed by the time-varying hFac)
u = UVELMASS.*areaWest_noh;
v = VVELMASS.*areaSouth_noh;
hConvV = (diff(u,1,1)+diff(v,1,2))./VVV;
% calculate vertical transport
w = WVELMASS.*areaTop;
if z(1)==1
 w(:,:,1) = 0;
end
vConvV = -diff(w,1,3)./VVV;




% OXYGEN CONCENTRATION BUDGET

rstarfac = repmat((1+ETAN_SNAP1./Depth),[1 1 nz]);

% read tracer
tmp = rdmds('diag_bgc',ts(t),'rec',1);
O2 = tmp(x,y,z);


% read O2 snapshots
tmp = rdmds('diag_bgc_snaps',[ts(t)-dts ts(t)],'rec',1);
O2_SNAP1 = tmp(x,y,z,1);
O2_SNAP2 = tmp(x,y,z,2);
% calculate tendency
dt = 86400*5;
tend_O2(:,:,:,t) = (O2_SNAP2−O2_SNAP1)/dt;

% read O2 flux
tmp = rdmds('diag_surf',ts(t),'rec',4);
O2FLUX = tmp(x,y);
% calculate surface forcing
surf_O2(:,:,:,t) = repmat(O2FLUX,[1 1 nz])./dzMat;
surf_O2(:,:,2:end,t) = 0;
surf_O2(:,:,:,t) = (−O2.*forcV+surf_O2(:,:,:,t))./rstarfac;

% read advective fluxes
advflux = rdmds('diag_o2_budget',ts(t),'rec',1:3);
advx = advflux([x x(end)+1],y,z,1);
advy = advflux(x,[y y(end)+1],z,2);
advz = advflux(x,y,[z z(end)+1],3);
% calculate horizontal advective divergence
adv_h_O2(:,:,:,t) = (diff(advx,1,1)+diff(advy,1,2))./VVV;
adv_h_O2(:,:,:,t) = (−O2.*hConvV+adv_h_O2(:,:,:,t))./rstarfac;
% top layer vertical advection is zero, not sure why
% replace by WTRAC03
tmp = rdmds('diag_o2_budget',ts(t),'rec',8);
advz(:,:,1) = tmp(x,y,1).*area;
% calculate vertical advective divergence
adv_v_O2(:,:,:,t) = -diff(advz,1,3)./VVV;
adv_v_O2(:,:,:,t) = (−O2.*vConvV+adv_v_O2(:,:,:,t))./rstarfac;

% read diffusive fluxes
difflux = rdmds('diag_o2_budget',ts(t),'rec',4:6);
difx = difflux([x x(end)+1],y,z,1);
dify = difflux(x,[y y(end)+1],z,2);
difz = difflux(x,y,[z z(end)+1],3);
% calculate horizontal advective divergence
dif_h_O2(:,:,:,t) = (diff(difx,1,1)+diff(dify,1,2))./VVV./rstarfac;
% calculate vertical advective divergence
dif_v_O2(:,:,:,t) = -diff(difz,1,3)./VVV./rstarfac;

% KPP
% kppO2 is 0 (seems like it's proportional to surface dilution term, which is 0 for O2)
% tmp = rdmds('diag_o2_budget',ts(t),'rec',7);
% kppO2(:,:,:,t) = -diff(tmp(x,y,[z z(end)+1]),1,3)./VVV./rstarfac;

% tendency due to biology
tmp = rdmds('diag_o2_budget',ts(t),'rec',10);
bio_O2(:,:,:,t) = tmp(x,y,z)./rstarfac;


end 



% patch edges

tend_O2(1128,:,:,:)=0;
tend_O2(:,336,:,:)=0;
tend_O2(:,:,51,:)=0;

surf_O2(1128,:,:,:)=0;
surf_O2(:,336,:,:)=0;
surf_O2(:,:,51,:)=0;

bio_O2(1128,:,:,:)=0;
bio_O2(:,336,:,:)=0;
bio_O2(:,:,51,:)=0;

adv_h_O2(1128,:,:,:)=0;
adv_h_O2(:,336,:,:)=0;
adv_h_O2(:,:,51,:)=0;

adv_v_O2(1128,:,:,:)=0;
adv_v_O2(:,336,:,:)=0;
adv_v_O2(:,:,51,:)=0;

dif_h_O2(1128,:,:,:)=0;
dif_h_O2(:,336,:,:)=0;
dif_h_O2(:,:,51,:)=0;

dif_v_O2(1128,:,:,:)=0;
dif_v_O2(:,336,:,:)=0;
dif_v_O2(:,:,51,:)=0;


% tend = forc-adv;
% res = tend+adv-forc;
res_O2 = tend_O2+adv_h_O2+adv_v_O2+dif_h_O2+dif_v_O2-surf_O2-bio_O2;

save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_tendO2_5day_2012.mat'],'tend_O2','time','lon','lat','depth');
save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_surfO2_5day_2012.mat'],'surf_O2','time','lon','lat','depth');
save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_bioO2_5day_2012.mat'],'bio_O2','time','lon','lat','depth');
save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_advhO2_5day_2012.mat'],'adv_h_O2','time','lon','lat','depth');
save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_advvO2_5day_2012.mat'],'adv_v_O2','time','lon','lat','depth');
save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_difhO2_5day_2012.mat'],'dif_h_O2','time','lon','lat','depth');
save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_difvO2_5day_2012.mat'],'dif_v_O2','time','lon','lat','depth');




x1 = 318; y1 = 111-37; t = 30;
figure;

clf
hold on;
plot(squeeze(tend_O2(x1,y1,:,t)));
plot(squeeze(surf_O2(x1,y1,:,t)));
plot(squeeze(adv_h_O2(x1,y1,:,t)+adv_v_O2(x1,y1,:,t)));
plot(squeeze(dif_h_O2(x1,y1,:,t)+dif_v_O2(x1,y1,:,t)));
plot(squeeze(bio_O2(x1,y1,:,t)));
plot(squeeze(res_O2(x1,y1,:,t)),'k:');




% BIOLOGICAL TERMS

for t=1:nt

display(t)

% read SSH snapshots
tmp = rdmds('diag_eta_snaps',ts(t)-dts,'rec',1);
ETAN_SNAP1 = tmp(x,y,1);

rstarfac = repmat((1+ETAN_SNAP1./Depth),[1 1 nz]);

% tendency due to biology
tmp = rdmds('diag_bio',ts(t),'rec',1:3);
NCP(:,:,:,t) = tmp(x,y,z,1)./rstarfac;
NPP(:,:,:,t) = tmp(x,y,z,2)./rstarfac;
Nfix(:,:,:,t) = tmp(x,y,z,3)./rstarfac;
REMIN(:,:,:,t) = NCP(:,:,:,t)-NPP(:,:,:,t)-Nfix(:,:,:,t);

end

NCP(1128,:,:,:)=0;
NCP(:,336,:,:)=0;
NCP(:,:,51,:)=0;

NPP(1128,:,:,:)=0;
NPP(:,336,:,:)=0;
NPP(:,:,51,:)=0;

Nfix(1128,:,:,:)=0;
Nfix(:,336,:,:)=0;
Nfix(:,:,51,:)=0;

REMIN(1128,:,:,:)=0;
REMIN(:,336,:,:)=0;
REMIN(:,:,51,:)=0;


save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_NCP_5day_2012.mat'],'NCP','time','lon','lat','depth');
save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_NPP_5day_2012.mat'],'NPP','time','lon','lat','depth');
save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_Nfix_5day_2012.mat'],'Nfix','time','lon','lat','depth');
save('-v7.3',['/data/SO6/TPOSE_diags/bgc_tpose6/2004_fwd_run/forYassir/tpfwd6_remin_5day_2012.mat'],'REMIN','time','lon','lat','depth');



x1 = 318; y1 = 111-37; t = 30;
figure;

O2toN  = 9.514;

clf
hold on;
plot(squeeze(bio_O2(x1,y1,:,t)));
plot(squeeze(NCP(x1,y1,:,t))*O2toN,'k:');
plot(squeeze(NPP(x1,y1,:,t))*O2toN);
plot(squeeze(Nfix(x1,y1,:,t))*O2toN);
plot(squeeze(REMIN(x1,y1,:,t))*O2toN);


