%function cyl_navier_stokes
clear; close all

%% Create mesh using distmesh2d (open source)

% mesh shape (distances)
xlimR = 10; % length of pipe
pipeR = 3; % radius of pipe
pipeLead = 3; % distance from inlet to centre of cylinder
cylR = 0.5; % radius of cylinder

% create function to define shape of mesh:
%  - shape of rectangle
%  - position and shape of cylinder
fd=@(p) ddiff(drectangle(p,-pipeLead,xlimR,-pipeR,pipeR),...
    dcircle(p,0,0,cylR));

% create function to define mesh refinement
r0 = 0.4;
fh=@(p) 0.3+min(1*(sqrt(sum(p.^2,2))-r0),2.5);

% create mesh using distmesh2d (open source package for meshing)
% nodePosition: position of nodes
% triIndex: indices for the triangles
figh = figure;
[nodePosition,triIndex]=distmesh2d(fd,fh,0.11,...
    [-pipeLead,-pipeR;xlimR,pipeR],...
    [-pipeLead,-pipeR;-pipeLead,pipeR;xlimR,-pipeR;xlimR,pipeR]);

%% Create mesh and reference element
% get:
%  - basis functions (Legendre polyonimals)
%  - quadrature rule to evaluate integrals (Gauss quadrature)
% use Taylor-Hood polynomials:
%  - velocity: order p polynomials
%  - pressure: order p-1 polynomials
% meshes{1}/refs{1}: x-velocity
% meshes{2}/refs{2}: y-velocity
% meshes{3}/refs{3}: pressure

% get mesh
[meshes,refs,ops] = make_mesh_cyl(nodePosition,triIndex,...
    xlimR,pipeR,pipeLead,cylR);

%% solve FE problem

% set plotting options (turn off to make code faster)
plotBool = false;

% set equation
dim = 2; eqn = CylFlow(dim); eqn.sym_adv = 1;
mmu{1} = 100; eqn.mu(1) = mmu{1}(1); eqn.steadyBool = true;

% get Stokes solution
mu0 = 0; eqn.mu(1) = mu0;
Ud = construct_dirichlet_constant_lift(eqn,meshes,refs,mu0);

% define inner product
ndof2 = size(meshes{1}.coord,1); ndof1 = size(meshes{3}.coord,1);
nnodes_p = (2*ndof2+1:ops.ndof-1)';
[Mx,Ax] = construct_masslap(eqn,meshes,refs);
ops.X = Mx + Ax; % use H1 norm for velocity
ops.X(nnodes_p,nnodes_p) = Mx(nnodes_p,nnodes_p); % Use L2 norm for pressure
ops.X(end,end) = 1;

% get interior and boundary nodes
bnodes = nodesonbgrp(meshes{1},1:5);
inodes = setdiff((1:ndof2)',bnodes);

% perturb the flow to create asymmetry
U1 = Ud; Up = 0.1*sin(pi*meshes{1}.coord(:,2)/pipeR);
U1(inodes) = U1(inodes) + Up(inodes);

% get functions for residual and mass matrix
eqn.mu(1) = mmu{1}(1);
resfun = @(U,t) compute_residual_vec(eqn,meshes,refs,ops,U,t);
Mtime = construct_mass(eqn,meshes,refs);

% time marching options (coarse time-step)
timeopt.order = 3; timeopt.h = 1; nt0 = 20;
timeopt.timedomain = [0,timeopt.h];
newtonopt = struct('verbosity',0,'maxiter',25);

% apply time-marching for FE solution (coarse time-step)
% Use coarse time-step to get initial condition
UU0 = zeros(length(U1), nt0); UU0(:,1) = U1;
for i = 2:nt0
    if (i <= timeopt.order)
        U0 = UU0(:,1);
    else
        U0 = UU0(:,i-1:-1:i-timeopt.order);
    end
    [~,Ui,~] = bdf(resfun,Mtime,U0,timeopt,newtonopt);
    UU0(:, i) = Ui(:,end);
    
    if (plotBool) % plot solutions (turn off to make code faster)
        figure(2), clf,
        title(sprintf('t = %.3e\n', timeopt.h*i));
        plotfield2d(meshes{1},refs{1},UU0(1:ndof2, i));
        axis equal;
        colorbar;
        drawnow;
        
        figure(1), clf,
        title(sprintf('t = %.3e\n', timeopt.h*i));
        plotfield2d(meshes{3},refs{3},UU0(2*ndof2 + (1:ndof1),i));
        axis equal;
        colorbar;
        drawnow
    end
end

% time marching options (finte time-step)
timeopt.order = 3; timeopt.h = 0.2; nt0 = 200;
timeopt.timedomain = [0,timeopt.h];
tt0 = (0:timeopt.h:(timeopt.h*(nt0-1)))';
newtonopt = struct('verbosity',0,'maxiter',25);

% apply time-marching for FE solution (fine time-step)
% Use fine time-step to get more accurate solutions
U0 = UU0(:,end); % get initial condition
UU0 = zeros(length(U1), nt0); UU0(:,1) = U0;
for i = 2:nt0
    if (i <= timeopt.order)
        U0 = UU0(:,1);
    else
        U0 = UU0(:,i-1:-1:i-timeopt.order);
    end
    [~,Ui,~] = bdf(resfun,Mtime,U0,timeopt,newtonopt);
    UU0(:, i) = Ui(:,end);
    
    if (plotBool) % plot solutions (turn off to make code faster)
        figure(2), clf,
        title(sprintf('t = %.3e\n', timeopt.h*i));
        plotfield2d(meshes{1},refs{1},UU0(1:ndof2, i));
        axis equal;
        colorbar;
        drawnow;
        
        figure(1), clf,
        title(sprintf('t = %.3e\n', timeopt.h*i));
        plotfield2d(meshes{3},refs{3},UU0(2*ndof2 + (1:ndof1),i));
        axis equal;
        colorbar;
        drawnow
    end
end

%% Compute lift and drag (FE)

% discard some data
iStart = 57;
UU = UU0(:,iStart:end); tt = tt0(iStart:end); nt = length(tt);

% compute lift and drag
[dragVec,liftVec] = compute_forces_fe(eqn,meshes,refs,ops,UU,tt);

% plot lift as a function of time
figure;
plot(liftVec(3:end),'-.'); hold on;
plot(dragVec(3:end),'-.');
legend("Lift","Drag",'location','best');
drawnow;

% find local max and min (to define one period)
tdxMax = find(islocalmax(liftVec));
ldx1 = tdxMax(end-1); ldx2 = tdxMax(end);

% compute mean lift and drag
dtp = tt(ldx2)-tt(ldx1-1);
meanLiftFE = sum(liftVec(ldx1:ldx2))*timeopt.h/dtp;
meanDragFE = sum(dragVec(ldx1:ldx2))*timeopt.h/dtp;

%% Get POD modes

% selection options for reduced basis
N = 3; % number of basis functions
shiftMode = true; % shift mode?

% get lifting function
if shiftMode
    % compute mean and steady solutions
    U_mean = mean(UU(:,ldx1:ldx2),2);
    resfun_s = set_resfun_vec(eqn,meshes,refs,ops);
    Us = newton(resfun_s,U_mean,newtonopt);
    U_lift = U_mean;
else
    U_lift = Ud;
end

% collect snapshots
UUpod = UU(:,ldx1:ldx2)-U_lift;

% POD for primal problem
Nmax = 20; podtol = 1e-8;
Zmax_pr = pod(ops.X,UUpod,Nmax,podtol);
Nmax_pr = size(Zmax_pr,2);

% get reduced basis
if shiftMode
    Npod = N-1;
    % compute shift mode
    Za = U_mean-Us;
    Zb = Za;
    for ndx = 1:Npod
        Zb = Zb-Za'*ops.X*Zmax_pr(:,ndx)*Zmax_pr(:,ndx);
    end
    Zs = Zb/sqrt(Zb'*ops.X*Zb);
    
    % assemble basis
    Z = [Zmax_pr(:,1:Npod),Zs];
else
    Z = Zmax_pr(:,1:N);
end

% set options for RB
rb_opt = struct('lifting','constant','Ud',U_lift);
rb_ops = construct_rb(eqn,meshes,refs,Z,rb_opt);

%% get coordinates for pressure sensors
% choose pressure sensors on surface of cylinder

% get nodes on surface of cylinder
cyl_nodes = nodesonbgrp(meshes{3},5);
cyl_coord = meshes{3}.coord(cyl_nodes,:);

% get coordinates for nodes we want to keep
% compute angle for each node, and organize by angle
thetaVec = sort(atan2(cyl_coord(:,2),cyl_coord(:,1)));
n_skip = 10; % number of nodes to skip
cyl_coord_sparse = cylR*[cos(thetaVec(1:n_skip:end)),...
    sin(thetaVec(1:n_skip:end-1))]; % compute cylinder coordinates

% get index for the cyl_coord_sparse (closest node to desired node)
cyl_nodes_sparse = zeros(length(cyl_coord_sparse),1);
for idx = 1:length(cyl_coord_sparse)
    thisDistance = vecnorm(cyl_coord-cyl_coord_sparse(idx,:),2,2);
    [~,thisIndex] = min(thisDistance);
    cyl_nodes_sparse(idx) = cyl_nodes(thisIndex);
end
cyl_nodes_sparse = unique(cyl_nodes_sparse);
np = length(cyl_nodes_sparse);

% update cyl_coord_sparse
cyl_coord_sparse = meshes{3}.coord(cyl_nodes_sparse,:);

% convert cyl_nodes_sparse to global
cyl_nodes_global = cyl_nodes_sparse + 1 + 2*ndof2;

% plot nodes of mesh
figure(3);
plot(meshes{3}.coord(:,1),meshes{3}.coord(:,2),'o');
axis equal; xlim([-pipeLead,xlimR]);ylim([-pipeR,pipeR]); hold on;
plot(cyl_coord(:,1),cyl_coord(:,2),'o');
plot(cyl_coord_sparse(:,1),cyl_coord_sparse(:,2),'x');
legend("All nodes","Cylinder nodes","Sparse cylinder nodes");
drawnow;

%% apply Ensemble Kalman filter to reduced-order model
% needs to be implemented

% get equations for RB problems in primal space
resfun_rb = @(U,t) compute_eqp_residual(eqn,rb_ops,U,t);
massfun_rb = @(U,t) compute_eqp_mass(eqn,rb_ops,U,t);

% get basis functions and lifting function to compute pressure
Z_sparse = Z(cyl_nodes_global,:);
U_lift_sparse = U_lift(cyl_nodes_global);

% solve RB problem (using time-marching)
a0 = Z'*ops.X*UUpod(:,1); % get initial RB coefficients using projection
timeopt_rb = timeopt; % copy time-marching structure
timeopt_rb.timedomain = [0,(nt-1)*timeopt_rb.h]; % update time domain
eqn.mu(1) = mmu{1}(1);
[~,aalpha_rb,~] = bdf(resfun_rb,massfun_rb,a0,timeopt_rb,newtonopt);

% set options for ENKF
gammaO = 1e-3; % covariance for observations
sigmaV = 1e-2; % covariance for state vector
IC_perturb = 1e-2;
H = Z_sparse; % linear observor operator
sScale = 1; % s in equation (10.11), noise or noise in observations
n_ens = 20; % set size of ensemble
beta = 1.01; %Anderson and Anderson inflation factor

% get initial condition (with noise)
a0 = aalpha_rb(:,1);
aalpha_enkf_mat = zeros(N,nt,n_ens);
for jdx = 1:n_ens
    thisXi = randn(N,1)*IC_perturb;
    aalpha_enkf_mat(:,1,jdx) = a0 + a0.*thisXi;
end

%TODO Encapsulate
Chat = zeros(N,N);
mhat = zeros(N,nt);
for i = 1:nt
    % prediction step
    for j = 1:n_ens
        % get solutions for time-marching
        if (i <= timeopt.order)
            a0 = aalpha_enkf_mat(:,1,j);
        else
            a0 = aalpha_enkf_mat(:,i-1:-1:i-timeopt.order,j);
        end
        % apply time-marching (update \hat{v}_{j+1})
        [~,ai,~] = bdf(resfun_rb,massfun_rb,a0,timeopt,newtonopt);
        ai_enkf(:,j) = ai(:,end);
    end
        %population mean
        mhat(:,i) = mean(ai_enkf,2); 
        
        % add noise to state vector
        thisXi = sigmaV*randn(N,n_ens);
        ai_enkf = mhat(:,i) + beta*(ai_enkf-mhat(:,i)) + thisXi;
            
        %Calcuate population covariance
        Chat=cov(ai_enkf');
        
        % apply observor operator (H\hat{v}_{j+1})
        pressure_predicted_i = H*ai_enkf+U_lift_sparse;
        
        % observe pressure (y_{j+1})
        thisEta = gammaO*randn(np,n_ens); % noise
        thisP = UU(cyl_nodes_global,i); % pressure
        pressure_observed_i = thisP + sScale*thisEta;
        
    % analysis step
    % Calculate Kalman matricies for update
    S = H*Chat*H'+gammaO*eye(np);
    K = Chat*H'/S;
    
    %Update time steps (\hat{v}_{j+1} \mapsto v_{j+1}
    %for j = 1:n_ens
    %    aij_enkf = aalpha_enkf_mat(:,i,j);
        aalpha_enkf_mat(:,i,:) = ai_enkf - K*(H*ai_enkf+U_lift_sparse) + K*pressure_observed_i;
    %end
end

% get EnKF RB coefficients
aalpha_enkf = mhat; % take ensemble mean
aalpha_rb_p = Z'*ops.X*(UU-U_lift); % get projected solution

% compute solutions in FE space
UU_enkf = Z*aalpha_enkf + U_lift;
UU_rb_p = Z*aalpha_rb_p + U_lift;
UU_rb = Z*aalpha_rb + U_lift;

sol_diff = UU-UU_enkf;
sol_diff_p = UU-UU_rb_p;
sol_diff_rb = UU-UU_rb;

% compute error (FE)
err_mat = zeros(nt,3);
for tdx = 1:size(UU,2)
    err_mat(tdx,1) = sqrt(sol_diff(:,tdx)'*ops.X*sol_diff(:,tdx));
    err_mat(tdx,2) = sqrt(sol_diff_rb(:,tdx)'*ops.X*sol_diff_rb(:,tdx));
    err_mat(tdx,3) = sqrt(sol_diff_p(:,tdx)'*ops.X*sol_diff_p(:,tdx));
end

% normalize error (FE)
U_mean = mean(UU,2);
fe_norm = sqrt(U_mean'*ops.X*U_mean);
err_mat = err_mat/fe_norm;

% compute mean solution (for normalization purposes)
a_mean = mean(aalpha_rb_p,2);

% compute error (RB)
err_mat_rb = zeros(nt,2);
for tdx = 1:nt
    err_mat_rb(tdx,1) = norm(aalpha_enkf(:,tdx)-aalpha_rb_p(:,tdx))/norm(aalpha_rb_p(:,tdx));
    err_mat_rb(tdx,2) = norm(aalpha_rb(:,tdx)-aalpha_rb_p(:,tdx))/norm(aalpha_rb_p(:,tdx));
end

% compute RMS
rms_mat = zeros(nt,1);
for tdx = 1:nt
    thisRMS = 0;
    for qdx = 1:n_ens
        thisRMS = thisRMS + norm(aalpha_enkf_mat(:,tdx,qdx)-aalpha_enkf(:,tdx))^2/norm(aalpha_enkf(:,tdx))^2;
    end
    rms_mat(tdx) = sqrt(thisRMS/(n_ens-1));
end

[dragVec_enkf,liftVec_enkf] = compute_forces_fe(eqn,meshes,refs,ops,UU_enkf,tt);
[dragVec_rb,liftVec_rb] = compute_forces_fe(eqn,meshes,refs,ops,UU_rb,tt);
[dragVec_rb_p,liftVec_rb_p] = compute_forces_fe(eqn,meshes,refs,ops,UU_rb_p,tt);

%calculate error in lift and drag estimates
dragerror_enkf=100*(dragVec_enkf-dragVec)./dragVec; 
dragerror_rb=100*(dragVec_rb-dragVec)./dragVec; 
dragerror_rb_p=100*(dragVec_rb_p-dragVec)./dragVec; 

lifterror_enkf=(liftVec_enkf-liftVec);
lifterror_rb=(liftVec_rb-liftVec); 
lifterror_rb_p=(liftVec_rb_p-liftVec); 
%% Plotting

% get time-step for plotting
tPlot = 0:timeopt.h:(timeopt.h*(nt-1));

% plot error (FE)
figure;
plot(tPlot,err_mat(:,1),'o',tPlot,err_mat(:,2),'o',tPlot,err_mat(:,3),'o');
xlabel("Time"); ylabel("Velocity Field L2 Error");
legend('EnKF','RB',"RB (projected)",'location','best');

% plot error (RB)
figure;
plot(tPlot,err_mat_rb(:,1),'o',tPlot,err_mat_rb(:,2),'o');
xlabel("Time"); ylabel("Reduced Basis Coefficient Relative Error (%)");
legend('EnKF','RB','location','best');

% plot RMS
figure;
plot(tPlot,rms_mat,'o');
xlabel("Time"); ylabel("RMS");

%plot force prediction error
figure;
plot(tt(2:end),dragerror_enkf,'o',tt(2:end),dragerror_rb,'o',tt(2:end),dragerror_rb_p,'o')
xlabel("Time"); ylabel("Drag Error (%)");
legend('EnKF','RB','RB (projected)')

figure;
plot(tt(2:end),lifterror_enkf,'o',tt(2:end),lifterror_rb,'o',tt(2:end),lifterror_rb_p,'o')
xlabel("Time"); ylabel("Lift Error");
legend('EnKF','RB','RB (projected)')
