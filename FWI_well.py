import os
import sys
import datetime
import numpy as np
import scipy.stats as st
import scipy.spatial as sp
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import scipy.interpolate as interpolate

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.interpolate as interpolate

fpth = os.path.abspath(os.path.join('rmwspy'))
sys.path.append(fpth)
from random_mixing_whittaker_shannon import *
from basics_fwi import *
from numpy import linalg as LA




class SonicWaveModel(NonLinearProblemTemplate):
    def __init__(self, domain, model, frequencies, Src_ids, Src_tags, Src_amps, use=None, data=None, communicator=None, testFieldConsistency=False):
        
        self.domain = domain
        self.model = model
        self.frequencies = frequencies
        self.Src_ids = Src_ids
        self.Src_tags = Src_tags
        self.Src_amps = Src_amps
        self.use = use   # mask which data from model output to be used
        self.data = data # the corresponding data
        self.communicator = communicator # MPI communicator, if None no MPI is used.
        if communicator is None:
            self.isMPIRoot = True
            self.useMPI=False
            self.testFieldConsistency
        else:
            self.isMPIRoot = communicator.Get_rank() == 0
            self.useMPI= communicator.Get_size() > 1
            self.testFieldConsistency=testFieldConsistency # this tests if all MPI ranks have the same values. Use this for debugging only. 
    
    def runFieldConsistencyTest(self, fields):
            """
            this checks fields as the same value on all MPI ranks.
            rank=0 is used as a reference
            """
            if self.communicator is not None:
                from mpi4py import MPI
                testfields=np.copy(fields)
                self.communicator.Bcast(testfields, root=0) 
                n=np.linalg.norm(fields)
                e=np.linalg.norm(fields-testfields)
                print("Consistence check: rank %s: difference of field to rank 0: %e (max=%e)."%(self.communicator.Get_rank(), e, n))
                if e < 1e-10* n:
                    errorcode=0
                else:
                    errorcode=1
                errorcode=self.communicator.allreduce(errorcode, op=MPI.MAX)
                if errorcode >0:
                    raise ValueError("Inconsistent random fields across ranks detected.")
                
    def objective_function(self, prediction):
        print("prediction dim=",prediction.shape)
        # return LA.norm((prediction-self.data)/self.data, axis=1)
        return LA.norm((np.log10(prediction/self.data)), axis=1)
        # if prediction.ndim == 1:
        #     return LA.norm(self.data - prediction)
        # elif prediction.ndim == 2:
        #     return LA.norm((self.data - prediction), axis=1)
        # elif prediction.ndim == 3:
        #     obs3d = np.atleast_3d(self.data).reshape(-1, 1, 1)
        #     return LA.norm((obs3d - prediction), axis=0)
    
    def allforwards(self, fields):
        """
        this runs all the forward models for the nfields realizations fields.
        and returns an array out[nfields,ndata] where ndata is the number of observations
        calculated by the forward model with out[i,:] being the output observations for
        field fields[i] (i=0,...,nfields-1)
        
        """
        ResultType=complex

        if self.testFieldConsistency: self.runFieldConsistencyTest(fields)
        nfields=fields.shape[0]
        nfrq=len(self.frequencies)
        ndata=self.data.shape[0]
        out = np.empty((nfields, ndata), dtype=ResultType)
        # this how it is done if there is no MPI involved:
        if not self.useMPI:
            for ifield in range(nfields):
                result=self.forward(self.frequencies, self.Src_amps, fields[ifield]) # this is for all frequencies, sources, observations
                out[ifield] = result[self.use]     # we grab all the observations we need marked by use
                # print('out1',out, flush=True)
        else:
            # this is shape of the return array of the model:
            data_shape=(len(self.Src_ids),  len(Receiver_ids))
            # this is portion of the work load of each rank:
            portionWork=(nfields*nfrq)//self.communicator.Get_size()
            if not (nfields*nfrq)%comm.Get_size() ==0 : portionWork+=1 # in case we have lost some work in the splitting
            # the results are first collected in this array:
            myResults=np.empty((portionWork,) +  data_shape, dtype=ResultType)
            for ifield in range(nfields):  # loop over fields
                for ifrq in range(nfrq):   # loop over frequency
                    i=ifield*nfrq+ifrq     # index of result in an virtual array of length (nfields*nfrq) 
                    irank=i//portionWork  # which rank should work on this index
                    if irank == self.communicator.Get_rank(): # if I am the rank, lets do it:
                        myResults[i%portionWork]=self.forward([ self.frequencies[ifrq] ], self.Src_amps[:,ifrq:ifrq+1], fields[ifield])[0]
            
            # we collect the big array of the results which is then copied to all ranks:
            results=np.empty( (self.communicator.Get_size(),portionWork) + data_shape, dtype=ResultType)
            self.communicator.Allgather(myResults, results)
            # now we need to remove the unused bits:
            results=results.reshape( (self.communicator.Get_size()*portionWork, )+data_shape)[:nfields*nfrq]
            # now this is reshaped to get the first dimension to be the number of data fields:
            results=results.reshape( (nfields,nfrq)+data_shape)
            # it assumed here that all ranks have the same values in the fields array:
            for ifield in range(nfields):
                out[ifield] = results[ifield][self.use]     # we grab all the observations we need marked by use
                # print('out2',out)
                # print('outshape=%s'%(out.shape,))
            # just checking Consistence:
            if self.testFieldConsistency: self.runFieldConsistencyTest(out)
        return out

    def marginal_transformation(self, T):
        return st.gamma.ppf(st.norm.cdf(T), 1.88, 1606, 309)
        # cdf = st.norm.cdf(T)
        # a = 1400
        # b = 2600
        # c = 2900
        # d = 4000
        # w2 = (d-c)/((d-c)+(b-a))
        # mask = cdf < 1-w2

        # cdf[mask] = st.uniform.ppf(cdf[mask], loc=a, scale=a+(b-a)/(1-w2)-a) 
        # cdf[~mask] = st.uniform.ppf(cdf[~mask], loc=d-((d-c)/w2), scale=d-(d-((d-c)/w2)) )

        # return cdf
        # return st.uniform.ppf(st.norm.cdf(T), 1400, 2700)

    def forward(self, frequencies, Src_amps, field):        
        # transform marginal
        t = self.marginal_transformation(field)
        t = np.fliplr(t)

        x = self.domain.getX()
              
        # if not self.useMPI: #this test can create chaos on MPI:
        #     assert inf(T)>0
        #     assert inf(S)>0
        if self.communicator:
            print("Rank: %d solves for frequencies %s"%(self.communicator.Get_rank(), frequencies))
        v=mapToDomain(domain, t, Resolution, origin=(PaddingX, PaddingZ))
        responses = self.model.runSurvey(frequencies, v, self.Src_ids, Src_amps, self.Src_tags)

        nlvals_at_x = responses # abs is just used in this pumping example
        # print("response", nlvals_at_x)
        return nlvals_at_x

def rSquare(estimations, measureds):
	""" Compute the coefficient of determination of random data. 
   This metric gives the level of confidence about the model used to model data"""
	SEE =  (abs( np.array(measureds) - np.array(estimations) )**2 ).sum()
	mMean = (np.array(measureds)).sum() / float(len(measureds))
	dErr = (abs(mMean - measureds)**2).sum()

	return 1 - (SEE/dErr)



if __name__ == "__main__":
    
    # import MPI module:
    # if failed some substitutes are set so this still runs on a single MPI rank
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        isMPIRoot = comm.Get_rank() == 0
        if isMPIRoot: print("MPI with %s ranks started."%comm.Get_size())
    except ImportError:
        comm=None
        isMPIRoot = True
    
    start = datetime.datetime.now()

    if isMPIRoot:
       ROOT='/scratch/uqacha25/interval_fwi_mpi'
       # mkDir(ROOT)
 
    import argparse

    parser = argparse.ArgumentParser(description='Ao\' cool RM FWI code')
    parser.add_argument('--nFields', '-n', metavar='nFields', dest='nFields', type=int, default=100, help='number of fields to simulate.')
    parser.add_argument('--start', '-s', metavar='START', dest='start', type=int, default=0, help='id number of first field.')
    parser.add_argument('--data', '-s', metavar='DATAFILE', dest='datafile', type=str, default='data.npy', help='data file (in, npy).')
    parser.add_argument('--const', '-c', metavar='CONSTFILE', dest='constrainsfile', type=str, default='newfield.npy', help='file of constraints (in, npy).')
    parser.add_argument('--real', '-r', metavar='REALFILE', dest='realizationfile', type=str, default='all_sim_data', help='file of realizations (out, npy).')
    parser.add_argument('--velo', '-r', metavar='VELOFILE', dest='velocityfile', type=str, default='field_velocity', help='file of velocities (out, npy).')
    parser.add_argument('--scatterplot', '-r', metavar='SCATTERPLOT', dest='scatterplot', type=str, default='scatter', help='files of sactter plots (out, png).')
    parser.add_argument('--veloplot', '-r', metavar='VELOPLOT', dest='velocityplot', type=str, default='velocity', help='files of velocity plots (out, png).')
    parser.add_argument('--calcmean', '-m' action='store_true', dest='calcmean', default=False,help='plot mean and std.')
    args = parser.parse_args()


    numWSNodes=8 # number of Whittaker-Shannon interpolation nodes (normally 8)

    # read survey:
    #survey=np.load("data(0930).npy")

    survey=np.load(args.datafile)
    Nx=survey.item().get('gridx')
    Ny=survey.item().get('gridz')
    Width=survey.item().get('width')
    Depth=survey.item().get('depth')
    # get data for inversion
    Source = survey.item().get('source_loc')
    src_tags=survey.item().get('sourcetags')
    Sourcetags = src_tags
    if isMPIRoot: print("%s Source found."%(len(Source)) )
    
    
    # same domain is generated on all MPI ranks:
    domain = Rectangle(Nx, Ny, l0=Width, l1=Depth, diracPoints=Source, diracTags=Sourcetags, order=1, fullOrder=True)
    # if isMPIRoot: print("Gmsh msh file read from %s"%(GMESHFN))
    X = domain.getX()
    # 
    model = SeismicWaveFrequency2DModel(domain)

    #pml
    Resolution = 10*U.m
    RefinementFactor = 2

    PaddingCellsX = 15   
    PaddingCellsZ = 10
    PaddingX = PaddingCellsX*Resolution
    PaddingZ = PaddingCellsZ*Resolution

    xmin = inf(X[0])
    xmax = sup(X[0])
    zmin = inf(X[1])
    zmax = sup(X[1])
    model.setpmlx(PaddingX,xmax)    
    model.setpmlz(PaddingZ,zmax)

    Receiver_ids = survey.item().get('nreceivers')
    Receiver_loc = survey.item().get('receiver')
    if isMPIRoot: print("Receiver :", Receiver_ids, "(", [ Receiver_loc[r] for r in Receiver_ids], ")")

    model.setReceivers(Receiver_loc, Receiver_ids)

    frequencies = survey.item().get('freq')
    Src_ids = survey.item().get('nsource')
    Src_tags = [ Sourcetags[s] for s in Src_ids]
    if isMPIRoot: print("Sources found  :", Src_tags, flush=True)
    Src_amps = survey.item().get('amplitude')
    # Src_amps=np.full((len(Src_tags), len(frequencies)),amps)
    Data = survey.item().get('signal') # This is just to make frequency the first dimension
    # print('Data.shape%s'%(Data.shape,))
    use = np.where(np.isnan(Data) == False)
    # print('use',use)
    data = Data[use]
    # print('data.shape%s'%(data.shape,))
    # print('data',data)
    D=[]
    R=[]

    numFrq = len(frequencies)
    if isMPIRoot: print("%s frequencies found."%numFrq)
    # Best efficiency is achieved when  numFrq * (numWSNodes-1) equals comm.Get_size() or is a multiple thereof.
    # (numWSNodes-1) is used as periodicity of interpolation nodes is use: 
    if comm is not None:
        if not (numFrq * (numWSNodes-1))%comm.Get_size() == 0:
            if isMPIRoot: print("INFORMATION: number of ranks (=%s) should be multiple of %s."%(comm.Get_size(),numFrq * (numWSNodes-1)))
        
    # initialize pumping model  
    my_model = SonicWaveModel(domain, model, frequencies=frequencies, Src_ids=Src_ids, Src_tags=Src_tags, Src_amps=Src_amps, use=use, data=data, communicator=comm, testFieldConsistency=False)
    ##
    #define linear constraints
    org_vel = np.load(args.constrainsfile)
    org_vel = np.fliplr(org_vel) # to get the correct values from that field
    x0 = org_vel.shape[0]
    z0 = org_vel.shape[1]
    x = x0*Resolution
    z = z0*Resolution

    nwell = 3
    step = 3
    well_axis = np.linspace(50, 150, nwell, dtype=int)
    hpos = np.arange(0, z0, step)
    well = []  
    

    for n in range (nwell):
        vpos = np.repeat(well_axis[n], hpos.shape[0])
        well.extend(vpos)
    well0 = np.array(well)
    well1 = np.tile(hpos,nwell)
    well_loc = np.vstack((well0,well1)).T
    well_sample = org_vel[well0[:], well1[:]]
    

##set up Random Mixing

    nFields = args.nFields
    cmod = '0.01 Nug(0.0) + 0.99 Sph(27)' 
    # This makes sure that all RMWSCondSim are identical. The seed argument needs to be the same on all MPI ranks.
    np.random.seed(345*(args.start+1)) 
    # define equality obsevations
    # observation point coordinates
    cp = np.array(well_loc)
    # cp2=np.array(well2_loc)
    # cp=np.concatenate((cp1,cp2))
    #observation values in actual data space
    cv = well_sample
    # cv =  org_vel[cp[:,0], cp[:,1]]
    #transform into standard normal space for the simulation 

    # a = 1400
    # b = 2600
    # c = 2900
    # d = 4100
    # w2 = (d-c)/((d-c)+(b-a))
    # mask = cv < c
    
    # cv[mask] = st.norm.ppf(st.uniform.cdf(cv[mask], loc=a, scale=d-c+b-a))
    # cv[~mask] = st.norm.ppf(st.uniform.cdf(cv[~mask], loc=c-b+a, scale=d-c+b-a ))

    cvn = st.norm.ppf(st.gamma.cdf(cv, 1.88, 1606, 309))
    # print('cvn', cvn)
    # initialize Random Mixing Whittaker-Shannon
    # INFO: this is running on all MPI ranks but only the fields generated on comm.Get_rank()==0 are used!
    CS = RMWSCondSim(my_model,
                    domainsize = (200, 50),
                    covmod = cmod,
                    nFields = nFields,
                    cp=cp,
                    cv=cvn,
                    p_on_circle = numWSNodes,
                    # optmethod = 'no_nl_constraints',
                    optmethod = 'circleopt',
                    minObj = 0.01,    
                    maxiter = 100,
                    maxbadcount=20,            # max number of consecutive iteration with less than frac_imp -> stopping criteria
                    frac_imp=0.9985, 
                    )

    # run RMWS
    CS()

    # save the fields:
    # note that they are in standard normal
    # To avoid overwriting the file this is done on the MPI root rank only:  
    if isMPIRoot:
        # np.save('sim_fields.npy', CS.finalFields)
        print('cs.fields',CS.finalFields)
    # to get a scatter plot of data vs sim 
    # we need to run the forward model again using them
    # Again this is only done on the MPI rank with comm.Get_rank()==0
    all_sim_data = my_model.allforwards(CS.finalFields)
    #np.save(os.path.join(ROOT,'all_sim_data33(0.005).npy'), all_sim_data)
    np.save((args.realizationfile+'{}.npy'.format(args.start)), all_sim_data)
    if isMPIRoot:
        for i in range(nFields):
            sim_data=all_sim_data[i]
            ###quantify the uncertainty through the R^2 value
            # x1=data
            # r_value=rSquare(sim_data, x1)
            # R.append(r_value)

            # plt.figure(figsize=(6,6))
            # plt.scatter(abs(data), abs(sim_data))
            # plt.plot(abs(x1), abs(x1), c='orange')
            # plt.xlim((0,16))
            # plt.ylim((0,16))
            # plt.xlabel('Observed data', fontsize=18)
            # plt.ylabel('Predicted data', fontsize=18)
            # plt.title(r'$R^2={:.3f}$'.format(r_value),fontsize=20)
            # plt.savefig(args.scatterplot+'{}.png'.format(i+args.start))
            # plt.clf()
            # plt.close()
	    
            # also plot the simulated field
            T = my_model.marginal_transformation(CS.finalFields[i])
            D.append(T)
            # np.save(os.path.join(ROOT,'field_velocity%i.npy'%i),T)
            plt.figure(figsize=(8,4))
            ax = plt.gca()
            im = ax.imshow(T.T,  cmap='jet', extent=(0,2000, 500, 0), vmin=1500, vmax=4000)
            ax.plot(cp[:,0]*Resolution, cp[:,1]*Resolution, '.', c='black', mew=2, ms=2)
            plt.gca().xaxis.set_ticks_position('top')
            plt.gca().xaxis.set_label_position('top')
            plt.xlabel('x (m)', fontsize=18)
            plt.ylabel('z (m)', fontsize=18)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.2)
            # plt.colorbar()
            clb = plt.colorbar(im, cax=cax)
            tick_locator = ticker.MaxNLocator(nbins=5)
            clb.locator = tick_locator
            clb.update_ticks()          
            plt.tight_layout()
            plt.savefig(args.velocityplot+'{}.png'.format(i+args.start),bbox_inches='tight', pad_inches=0)
            plt.clf()
            plt.close()
        D = np.array(D)
        # np.save(os.path.join(ROOT,'R33.npy'),R)
        np.save(args.velocityfile+'{}.npy'.format(args.start),D)
	
## quantify the uncertainty via mean and standard deviation
        # if args.calcmean:
        #     mean = np.mean(D, axis=0)
        #     # np.save(os.path.join(ROOT,'mean0819.npy'), mean.T)
        #     plt.figure(figsize=(8,4))
        #     ax = plt.gca()
        #     # plt.title("Mean")
        #     im = ax.imshow(mean.T,  cmap='jet', extent=(0,2000, 500, 0), vmin=1500, vmax=4000)
        #     ax.plot(cp[:,0]*Resolution, cp[:,1]*Resolution, '.', c='black', mew=2, ms=2)
        #     plt.gca().xaxis.set_ticks_position('top')
        #     plt.gca().xaxis.set_label_position('top')
        #     plt.xlabel('x (m)', fontsize=18)
        #     plt.ylabel('z (m)', fontsize=18)
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="2%", pad=0.2)
        #     clb = plt.colorbar(im, cax=cax)
        #     tick_locator = ticker.MaxNLocator(nbins=5)
        #     clb.locator = tick_locator
        #     clb.update_ticks()          
        #     plt.tight_layout()
        #     plt.savefig('mean33(0.005).png',bbox_inches='tight', pad_inches=0)
        #     plt.clf()
        #     plt.close()


        #     sta_dev = np.std(D,axis=0)
        #     # np.save(os.path.join(ROOT,'sta_dev0819.npy'), sta_dev.T)
        #     # print("Standard Deviation of sample is % s " % sta_dev) 
        #     plt.figure(figsize=(8,4))
        #     ax = plt.gca() 
        #     # plt.title("Standard Deviation", fontsize=14)
        #     im = ax.imshow(sta_dev.T, cmap='seismic', extent=(0,2000, 500, 0))
        #     plt.plot(cp[:,0]*Resolution, cp[:,1]*Resolution, '.', c='black', mew=2, ms=2)
        #     plt.gca().xaxis.set_ticks_position('top')
        #     plt.gca().xaxis.set_label_position('top')
        #     plt.xlabel('x (m)', fontsize=18)
        #     plt.ylabel('z (m)', fontsize=18)
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="2%", pad=0.2)
        #     clb = plt.colorbar(im, cax=cax)
        #     tick_locator = ticker.MaxNLocator(nbins=5)
        #     clb.locator = tick_locator
        #     clb.update_ticks()          
        #     plt.tight_layout()
        #     plt.savefig('standard deviation33(0.005).png',bbox_inches='tight', pad_inches=0)
        #     plt.clf()
        #     plt.close()


    end = datetime.datetime.now()
    print('time needed:', end - start)









