#===============================================================================
# DESCRIPTION
#    Detect the cold front
#===============================================================================




   
#===============================================================================
# Detect vold front
#===============================================================================

if __name__ =='__main__':
    P = net_LCB.getvarallsta("Pa H", ['C09'],by='H')
    P = P.resample('1D').mean()
    P_roll = P.rolling(30, center=True).mean()
#     P_roll.plot()2
#     P.plot()
#     plt.show()
    plt.plot(P_roll)
    plt.plot(P)
    plt.show()
      
    P = P-P_roll
#     P = P.diff()
    plt.plot(P)
      
    front = P[P < -4].dropna()
    print front
#     P.plot()
    plt.plot(front,marker='o')
# #     P.plot()
    plt.show()
    front_date = front.index
#     prefront = front_date
#     postfront = front_date
      
    for i,date in enumerate(front_date):
        if i ==0:
            prefront = pd.date_range(date - timedelta(days=3), date- timedelta(days=0), freq='H')
            postfront = pd.date_range(date + timedelta(days=0), date+ timedelta(days=3), freq='H')
        else:
            prefront = prefront.append(pd.date_range(date - timedelta(days=3), date- timedelta(days=0), freq='H'))
            postfront = postfront.append(pd.date_range(date + timedelta(days=0), date+ timedelta(days=3), freq='H'))
      
    prefront = prefront.sort_values()
#     prefrontmask = np.isfinite(prefront)
      
    postfront = postfront.sort_values()
#     postfrontmask = np.isfinite(postfront)
    print front_date
  
  
    hour='21:00'
    plt.plot( error.loc[prefront, 'C09'].between_time(hour, hour),'o', c='b', label='prefrontal')
    plt.plot(error.loc[postfront, 'C09'].between_time(hour, hour),'o',c='r', label='postfrontal')
#     plt.plot(error.loc[front_date, 'C09'].between_time(hour, hour),'o',c='r', label='frontal')
    plt.legend()
    plt.show()
      
  
