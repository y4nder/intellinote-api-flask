context = """ In automata theory, a key distinction is made between deterministic and non-deterministic automata. 
    A **Deterministic Finite Automaton (DFA)** is an automaton where for each state and input symbol, 
    there is exactly one transition to another state. This means that given a specific input string, 
    a DFA will always process the string in the same way, producing a unique result. 
    DFAs are particularly useful because they can be implemented in a straightforward way using finite state machines, 
    and they recognize regular languages, which are crucial in text processing, pattern matching, and lexical analysis.
    On the other hand, a **Non-Deterministic Finite Automaton (NFA)** is an automaton where for some state and input symbol, 
    there may be multiple possible transitions to different states. This non-determinism means that an NFA can explore 
    multiple computation paths simultaneously. While an NFA might seem more powerful, it is not more expressive than a DFA 
    in terms of the languages they can recognize—both can recognize exactly the same class of languages, called regular languages. 
    However, NFAs are more flexible and can often be easier to design in cases where a DFA would require more states. 
    The process of converting an NFA to a DFA, known as the powerset construction or subset construction algorithm, shows that 
    while NFAs can be more compact in their design, they may require more states in the corresponding DFA.
    Understanding the difference between deterministic and non-deterministic automata is vital for the study of computational 
    complexity and the development of efficient algorithms in areas such as compiler construction, network protocols, and artificial intelligence."""
    
    
def get_sample_note() -> str:
    return context