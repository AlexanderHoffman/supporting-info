#!/usr/bin/env python

#Auxiliary file for vasp_freq.py

# -*-coding:utf-8 -*-

"""
module myxml
============
 
Description
-----------

Ce module contient quelques fonctions simples pour lire un fichier xml. Les fonctions
permettent à partir d'une liste de ligne xml ou d'un fichier xml, d'extraire un bloc
de données contenu entre deux balises, d'obtenir les attributs et les valeurs associées à
une balise, d'extraire les données entre deux balises et d'obtenir le nom d'une
balise.

Vocabulaire
-----------

Par la suite on emploiera les termes "balise", "clefs", "valeur" et "bloc" avec le
sens suivant :

<balise clefs1="valeur" clefs2="valeur">
    bla bla
</balise>

Dans les exemples, on travaille sur un fichier xml comme celui ci :
<monxml>
    <info>
        <i> coucou </i>
        <age nom="toto">  18 </age>
        <taille nom="tata"> 160 </taille>
    </info>
    <param>
        <allsets>
            <set info="att 1" type="a">
                <i> 1. </i>
            </set>
            <set info="att 2" type="b">
                <i> 2. </i>
            </set>
            <set info="att 3" type="a">
                <i> 3. </i>
            </set>
        </allsets>
    </param>
</monxml>

"""

xml = """<monxml>
    <info>
        <i> coucou </i>
        <age nom="toto">  18 </age>
        <taille nom="tata"> 160 </taille>
    </info>
    <param>
        <allsets>
            <set info="att 1" type="a">
                <i> 1. </i>
            </set>
            <set info="att 2" type="b">
                <i> 2. </i>
            </set>
            <set info="att 3" type="a">
                <i> 3. </i>
            </set>
        </allsets>
    </param>
</monxml>""".split("\n")

import doctest

__licence__ = "GPL"
__author__ = "Germain Vallverdu <germain.vallverdu@univ-pau.fr>"

def getBlocs(balise, liste, clefs = None, onlyfirst = False):
    """ Renvoie une liste dont chaque élément contient la liste des lignes correspondantes
        aux blocs délimités par balise, c'est à dire comprises entre <balise> et
        </balise>.

        Arguments :
            * balise    : string, nom de la balise
            * liste     : list ou file, données xml à traiter
            * clefs     : dict, dictionnaire de {clefs: valeur}
            * onlyfirst : bool

        Exemples :

        On récupère le bloc définie par la balise info
        >>> getBlocs("info", xml)[0]
        ['<info>', '<i> coucou </i>', '<age nom="toto">  18 </age>', '<taille nom="tata"> 160 </taille>', '</info>']

        On récupère les blocs définis par la balise set
        >>> getBlocs("set", xml)
        [['<set info="att 1" type="a">', '<i> 1. </i>', '</set>'], ['<set info="att 2" type="b">', '<i> 2. </i>', '</set>'], ['<set info="att 3" type="a">', '<i> 3. </i>', '</set>']]

        On récupère le premier bloc défini par la balise set
        >>> getBlocs("set", xml, onlyfirst = True)
        ['<set info="att 1" type="a">', '<i> 1. </i>', '</set>']

        On récupère le bloc défini par la balise set avec le clefs type égale à "b" et la
        clefs info égale à "att 2"
        >>> getBlocs("set", xml, {"type":"b", "info":"att 2"})
        [['<set info="att 2" type="b">', '<i> 2. </i>', '</set>']]

        On récupère les blocs définis par la balise set avec la clefs type égale à "a"
        >>> getBlocs("set", xml, {"type":"a"})
        [['<set info="att 1" type="a">', '<i> 1. </i>', '</set>'], ['<set info="att 3" type="a">', '<i> 3. </i>', '</set>']]

    """

    # defini la variable de retour comme nulle
    blocs = list()

    # delimiteurs du bloc
    openbloc  = "<" + balise.strip()
    closebloc = "</" + balise.strip() + ">"

    # liste des lignes en retour
    start = False
    nbloc = 0
    for ligne in liste:
        if openbloc in ligne:
            if not start:
                if clefs is not None:
                    # cherche les attributs
                    valeursAttributs = getClefs( ligne, clefs.keys() )

                    # verifie la valeur des attributs
                    for att in clefs.keys():
                        if clefs[ att ] == valeursAttributs[ att ]:
                            start = True
                        else:
                            break

                    if start:
                        nopen = 1
                        nclose = 0
                        nbloc += 1
                        blocs.append( list() )

                else :
                    nopen = 1
                    nclose = 0
                    start = True
                    nbloc += 1
                    blocs.append( list() )

            else:
                nopen += 1

        if closebloc in ligne and start == True : 
            blocs[nbloc - 1].append(ligne.strip())
            nclose += 1
            if nclose == nopen:
                start = False
                if onlyfirst:
                    break
            else:
                continue

        if start : 
            blocs[nbloc - 1].append(ligne.strip())

    # return None if balise not found
    if len(blocs) == 0:
        blocs = None

    # si un seul bloc demandé, retourne la liste des lignes uniquement
    if onlyfirst and blocs != None:
        blocs = blocs[0]

    return blocs

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

def getClefs(ligne, listeClefs):
    """ Retourne dans un dictionnaire, les valeurs des clefs indiquées dans ``listeClefs``
    lues sur ``ligne``. Si une clefs n'est pas trouvée la valeur None lui est attribuée.

        Arguments :
            * ligne      : string, ligne xml
            * listeClefs : list, liste des clefs dont on veut les valeurs

        Exemple :
        >>> ligne = '<i type="string" name="PREC">medium</i>'
        >>> getClefs(ligne, ["type", "name"])
        {'type': 'string', 'name': 'PREC'}

    """

    clefs = dict()

    for att in listeClefs:
        pos = ligne.find( str(att) )
        if pos == -1:
            clefs[str(att)] = None
        else :
            k = ligne[pos:].find("\"")
            k += pos + 1
            l = ligne[k:].find("\"")
            l += k 

            if l < k:
                clefs[str(att)] = None
            else:
                clefs[ str(att) ] = ligne[k:l]

    return clefs

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

def getNodeData(ligne):
    """ Retourne les valeurs entre deux balises xml sur ligne  

        Arguments :
            * ligne : string, ligne xml

        Exemples :
        >>> ligne = '<i type="string" name="PREC">medium</i>'
        >>> getNodeData(ligne)
        ['medium']

        >>> ligne = '<i name="subversion" type="string">3Apr08 complex  serial</i>'
        >>> getNodeData(ligne)
        ['3Apr08', 'complex', 'serial']

        >>> ligne = '<i name="ENCUT">    450.00000000</i>'
        >>> getNodeData(ligne)
        ['450.00000000']
    
    """

    debutValeurs = False
    tmpValeurs = list()
    for c in ligne:
        if c == "<": 
            debutValeurs = False
            continue

        if c == ">": 
            debutValeurs = True
            continue

        if debutValeurs : 
            tmpValeurs.append(c)

    valeurs = "".join(tmpValeurs)

    return valeurs.split()

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

def getNodeName( ligne ) :
    """ retourne le nom de la balise sur une ligne d'un fichier xml 
    
        Arguments :
            * ligne : string, ligne xml

        Exemples :
        >>> ligne = "<kpoints></kpoints>"
        >>> getNodeName(ligne)
        'kpoints'
        
        >>> ligne = '<i name="subversion" type="string">3Apr08 complex  serial</i>'
        >>> getNodeName(ligne)
        'i'

        >>> ligne = '<  generation   param="listgenerated">'
        >>> getNodeName(ligne)
        'generation'

    """

    k = ligne.find("<")
    l = ligne.find(">")
    return ligne[k+1:l].split()[0].strip()

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

if __name__ == "__main__":
    print(__doc__)
    print("LANCEMENT DES TESTS :")
    print("    doctest.testmod()\n")
    tests = doctest.testmod()
    print(tests)
    print("\nfin")
