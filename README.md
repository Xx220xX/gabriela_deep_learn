# gabriela_deep_learn
It is really amazing to get a library with this performance.

For a celeron 1.1 GHz processor the feed and forward time for an example was 100 ms,for an architecture (2,1400,800,240,28,1).

so far no test has been done with a gpu or a more powerful processor

### Why is gabriela so fast?
optimized matrix multiplication methods are used. This is Gabriela's big secret.

# Get start
to start, import the module gabriela a class DNNto start, import the module gabriela a class DNN 
```python
from grabriela import DNN
```
create an instance for class DNN

you must pass as an argument an architecture for your network.


```python
dnn = DNN((2,6,5,4,1))
```

