from matplotlib import pyplot as plt
import json
import numpy as np

if __name__ == "__main__":
    results_conditions = json.load(open("annotation_round_1/adjudication/agreement_conditions.json"))
    
    # Conditions metrics
    total_conditions = results_conditions["n_conditions"]
    
    INTERESTING_METRICS = [
        "evidence_support",
        "meaningfulness",
        "atomicity",
        "type_validity",
        "non_redundancy"
    ]
    
    # 1. MODIFICATO FIGSIZE: Ora più largo che alto per ospitare 3 colonne affiancate
    plt.figure(figsize=(18, 11)) 
    
    index = 1
    for d in results_conditions:
        if d in INTERESTING_METRICS and isinstance(results_conditions[d], dict):
            # Layout aggiornato a 2 righe, 3 colonne
            ax = plt.subplot(2, 3, index)
            
            # Garantisce che il grafico a torta sia perfettamente circolare
            ax.axis('equal')

            # Titolo del subplot
            title_text = d.replace("_", " ").capitalize()
            ax.set_title(title_text, weight='bold', pad=25, fontsize=12)

            major_disagreement = results_conditions[d]["major_disagreement"]
            minor_disagreement = results_conditions[d]["minor_disagreement"]
            agreement = total_conditions - major_disagreement - minor_disagreement

            labels = ["Agreement", "Minor disagreement", "Major disagreement"]
            values = [agreement, minor_disagreement, major_disagreement]
            
            # Palette ad alto contrasto (Color-blind friendly)
            colors = ["#0072B2", "#E69F00", "#4A4A4A"]

            # Disegno della ciambella
            wedges, texts = ax.pie(
                values,
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.45, edgecolor='w', linewidth=1.5)
            )

            # Stile della freccia lineare
            kw = dict(
                arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.8, connectionstyle="arc3,rad=0"), 
                zorder=2, 
                va="center"
            )

            for i, p in enumerate(wedges):
                # Esclude i valori a zero per non sovrapporre frecce
                if values[i] == 0:
                    continue
                
                # Calcolo geometrico dell'angolo
                ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                
                horizontalalignment = "left" if x > 0 else "right"
                
                pct = (values[i] / total_conditions) * 100
                label_text = f"{labels[i]}\n{pct:.1f}% (n={values[i]})"
                
                # Proiezione radiale (lunghezza 1.15 per dare un filo di respiro in più nelle colonne)
                arrow_length = 1.15
                text_x = arrow_length * x
                text_y = arrow_length * y
                
                ax.annotate(
                    label_text, 
                    xy=(x, y), 
                    xytext=(text_x, text_y),
                    horizontalalignment=horizontalalignment, 
                    fontsize=9.5,
                    linespacing=1.2,
                    **kw
                )

            index += 1

    # 2. FIX SUBPLOT VUOTO: Aggiornato alla griglia 2x3 per nascondere correttamente il sesto quadrante
    if index == 6:
        ax_empty = plt.subplot(2, 3, 6)
        ax_empty.axis('off')

    # 3. FIX SPAZIATURA: Aumentati i margini laterali (left/right) per evitare che i testi 
    # delle colonne esterne vengano tagliati o si avvicinino troppo al bordo della pagina PDF
    plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.10, wspace=0.55, hspace=0.45)
    
    # Salvataggio ad altissima qualità vettoriale
    plt.savefig(
        "figures/agreement/agreement_conditions.pdf", 
        bbox_inches="tight", 
        pad_inches=0.4,
        backend="pdf",
        dpi=300
    )


    # descriptions metrics
    results_descriptions = json.load(open("annotation_round_1/adjudication/agreement_descriptions.json"))
    total_descriptions = results_descriptions["n"]
    
    INTERESTING_METRICS = [
        "description_faithfulness",
        "description_evidence_support",
        "description_specificity",
        "unsupported_claims"
    ]
    
    # 1. MODIFICATO FIGSIZE: Ora più largo che alto per ospitare 3 colonne affiancate
    plt.figure(figsize=(18, 11)) 
    index = 1
    for d in results_descriptions:
        if d in INTERESTING_METRICS and isinstance(results_descriptions[d], dict):
            ax = plt.subplot(2, 2, index)
    
            # Garantisce che il grafico a torta sia perfettamente circolare
            ax.axis('equal')
    
            # Titolo del subplot
            title_text = d.replace("_", " ").replace("description", "").capitalize()
            ax.set_title(title_text, weight='bold', pad=25, fontsize=12)
    
            major_disagreement = results_descriptions[d]["major_disagreement"]
            minor_disagreement = results_descriptions[d]["minor_disagreement"]
            agreement = total_descriptions - major_disagreement - minor_disagreement
    
            labels = ["Agreement", "Minor disagreement", "Major disagreement"]
            values = [agreement, minor_disagreement, major_disagreement]
    
            # Palette ad alto contrasto (Color-blind friendly)
            colors = ["#0072B2", "#E69F00", "#4A4A4A"]
    
            # Disegno della ciambella
            wedges, texts = ax.pie(
                values,
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.45, edgecolor='w', linewidth=1.5)
            )
    
            # Stile della freccia lineare
            kw = dict(
                arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.8, connectionstyle="arc3,rad=0"), 
                zorder=2, 
                va="center"
            )
    
            for i, p in enumerate(wedges):
                # Esclude i valori a zero per non sovrapporre frecce
                if values[i] == 0:
                    continue
    
                # Calcolo geometrico dell'angolo
                ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
    
                horizontalalignment = "left" if x > 0 else "right"
    
                pct = (values[i] / total_descriptions) * 100
                label_text = f"{labels[i]}\n{pct:.1f}% (n={values[i]})"
    
                # Proiezione radiale (lunghezza 1.15 per dare un filo di respiro in più nelle colonne)
                arrow_length = 1.15
                text_x = arrow_length * x
                text_y = arrow_length * y
    
                ax.annotate(
                    label_text, 
                    xy=(x, y), 
                    xytext=(text_x, text_y),
                    horizontalalignment=horizontalalignment, 
                    fontsize=9.5,
                    linespacing=1.2,
                    **kw
                )
    
            index += 1
    
    # 2. FIX SUBPLOT VUOTO: Aggiornato alla griglia 2x3 per nascondere correttamente il sesto quadrante
    if index == 6:
        ax_empty = plt.subplot(2, 3, 6)
        ax_empty.axis('off')
    
    # 3. FIX SPAZIATURA: Aumentati i margini laterali (left/right) per evitare che i testi 
    # delle colonne esterne vengano tagliati o si avvicinino troppo al bordo della pagina PDF
    plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.10, wspace=0.55, hspace=0.45)
    
    # Salvataggio ad altissima qualità vettoriale
    plt.savefig(
        "figures/agreement/agreement_descriptions.pdf", 
        bbox_inches="tight", 
        pad_inches=0.4,
        backend="pdf",
        dpi=300
    )