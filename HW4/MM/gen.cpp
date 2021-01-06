#include<iostream>
using namespace std;


int main() {
    int m, n, l; 
    cin >> n >> m >> l;

    cout << n << ' ' << m << ' ' << l << '\n';

    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            cout << i+j << " \n"[ j+1 == m];
        }
    }
    for(int i=0; i<m; i++) {
        for(int j=0; j<l; j++) {
            cout << i+j << " \n"[ j+1 == l];
        }
    }
}